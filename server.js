// ─────────────────────────────────────────────────────────────────
// backend/server.js
//
// Relay server between Angular browser client and OpenAI Realtime API.
//
// WHY A BACKEND IS NEEDED:
//   The OpenAI Realtime API requires a WebSocket connection with the
//   API key in the header. Browsers cannot set custom WebSocket headers.
//   So we proxy: Browser <─WS─> This server <─WS─> OpenAI Realtime
//
// WHAT THIS SERVER DOES:
//   1. Accepts WebSocket from Angular with { type: 'init', resumeText, skills }
//   2. Opens a WebSocket to OpenAI Realtime with your API key
//   3. Sends session.update to configure VAD, voice, system prompt
//   4. Forwards raw audio chunks from browser → OpenAI
//   5. Forwards ALL OpenAI events back to browser
//      (browser handles: transcript, text, audio, done events)
//
// OPENAI REALTIME VAD:
//   OpenAI's server_vad detects speech boundaries using acoustic models
//   trained on millions of hours of speech. It fires:
//     input_audio_buffer.speech_started  → user started speaking
//     input_audio_buffer.speech_stopped  → user stopped speaking
//     input_audio_buffer.committed       → audio sent for transcription
//     conversation.item.input_audio_transcription.completed → transcript ready
//   This is NEURAL VAD — no volume thresholds, no timers.
// ─────────────────────────────────────────────────────────────────

import { WebSocketServer, WebSocket } from 'ws';
import dotenv from 'dotenv';

dotenv.config();

const PORT       = process.env.PORT || 3001;
const OPENAI_KEY = process.env.OPENAI_API_KEY;

if (!OPENAI_KEY) {
  console.error('❌ OPENAI_API_KEY missing from .env');
  process.exit(1);
}

const wss = new WebSocketServer({ port: PORT });
console.log(`✅ TalentAI relay server running on ws://localhost:${PORT}`);

// ── Per-connection handler ────────────────────────────────────────
wss.on('connection', (browserWs) => {
  console.log('🔌 Browser client connected');

  let openaiWs    = null;   // WebSocket to OpenAI Realtime
  let sessionReady = false; // true after session.created received
  let resumeText   = '';
  let skills       = '';
  let questionCount = 0;    // track questions for prompt adaptation

  // ── Handle messages from Angular browser ─────────────────────
  browserWs.on('message', (data, isBinary) => {

    // Binary = raw PCM16 audio chunk from mic
    if (isBinary) {
      if (openaiWs?.readyState === WebSocket.OPEN) {
        // Forward raw audio to OpenAI input buffer
        const b64 = Buffer.from(data).toString('base64');
        openaiWs.send(JSON.stringify({
          type: 'input_audio_buffer.append',
          audio: b64,
        }));
      }
      return;
    }

    // Text = JSON control messages
    let msg;
    try {
      msg = JSON.parse(data.toString());
    } catch {
      return;
    }

    // ── Init message: browser sends resume context ────────────
    if (msg.type === 'init') {
      resumeText    = msg.resumeText  || '';
      skills        = msg.skills      || '';
      questionCount = msg.questionCount || 0;

      // Open connection to OpenAI Realtime API
      openaiWs = new WebSocket(
        'wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview-2024-12-17',
        {
          headers: {
            'Authorization': `Bearer ${OPENAI_KEY}`,
            'OpenAI-Beta':   'realtime=v1',
          },
        }
      );

      openaiWs.on('open', () => {
        console.log('✅ Connected to OpenAI Realtime');

        // ── Configure the session ─────────────────────────────
        // This is the most important message — sets VAD, voice,
        // instructions, and audio format
        openaiWs.send(JSON.stringify({
          type: 'session.update',
          session: {

            // ── Modalities: audio + text ──────────────────────
            modalities: ['audio', 'text'],

            // ── System instructions (interviewer persona) ─────
            instructions: buildInterviewerPrompt(resumeText, skills, questionCount),

            // ── Voice: shimmer is warm, closest Indian-EN ─────
            voice: 'shimmer',

            // ── Audio format ──────────────────────────────────
            // pcm16 at 24kHz for both input and output
            // Browser must send PCM16 @ 24kHz (we handle this in frontend)
            input_audio_format:  'pcm16',
            output_audio_format: 'pcm16',

            // ── Transcription model ───────────────────────────
            // whisper-1 gives accurate transcripts including fillers
            input_audio_transcription: {
              model: 'whisper-1',
            },

            // ── SERVER-SIDE VAD CONFIGURATION ─────────────────
            // This is the entire reason we're using Realtime API.
            // OpenAI's acoustic model detects speech boundaries.
            // No volume thresholds. No timers. Neural detection.
            turn_detection: {
              type: 'server_vad',

              // How sensitive the VAD is (0.0–1.0)
              // 0.5 is default. Lower = more sensitive (picks up soft voice)
              // Higher = less sensitive (ignores background noise)
              threshold: 0.4,

              // How much audio before speech counts (ms)
              // Prevents single-word fragments from triggering
              prefix_padding_ms: 300,

              // How long silence AFTER speech before committing (ms)
              // 800ms = natural conversational pause without cutting off
              // This is tuned for interview answers (slightly longer than chat)
              silence_duration_ms: 800,
            },

            // ── Response config ───────────────────────────────
            // Temperature 0.7 = professional but not robotic
            // Max output tokens capped at 150 = concise questions
            temperature: 0.7,
            max_response_output_tokens: 150,
          },
        }));
      });

      // ── Handle events from OpenAI → forward to browser ───────
      openaiWs.on('message', (rawEvent) => {
        let event;
        try {
          event = JSON.parse(rawEvent.toString());
        } catch {
          return;
        }

        // Log key events (remove in production)
        if (!['response.audio.delta'].includes(event.type)) {
          console.log('OpenAI →', event.type);
        }

        switch (event.type) {

          // ── Session ready ───────────────────────────────────
          case 'session.created':
          case 'session.updated':
            sessionReady = true;
            browserWs.send(JSON.stringify({ type: 'ready' }));
            break;

          // ── VAD: user started speaking ──────────────────────
          case 'input_audio_buffer.speech_started':
            browserWs.send(JSON.stringify({ type: 'user_speech_start' }));
            break;

          // ── VAD: user stopped speaking ──────────────────────
          case 'input_audio_buffer.speech_stopped':
            browserWs.send(JSON.stringify({ type: 'user_speech_stop' }));
            break;

          // ── VAD committed the audio for transcription ───────
          case 'input_audio_buffer.committed':
            browserWs.send(JSON.stringify({ type: 'user_audio_committed' }));
            break;

          // ── Live transcript of what user said ───────────────
          case 'conversation.item.input_audio_transcription.completed':
            browserWs.send(JSON.stringify({
              type:       'user_transcript',
              transcript: event.transcript,
              item_id:    event.item_id,
            }));
            break;

          // ── Alex's response text streaming ──────────────────
          case 'response.text.delta':
            browserWs.send(JSON.stringify({
              type:  'response_text_delta',
              delta: event.delta,
            }));
            break;

          // ── Alex's response text complete ───────────────────
          case 'response.text.done':
            browserWs.send(JSON.stringify({
              type: 'response_text_done',
              text: event.text,
            }));
            questionCount++;
            break;

          // ── Alex's audio streaming (PCM16 chunks) ───────────
          // This is the hot path — fires many times per second
          case 'response.audio.delta':
            browserWs.send(JSON.stringify({
              type:  'response_audio_delta',
              delta: event.delta, // base64 PCM16 chunk
            }));
            break;

          // ── Alex's audio complete ────────────────────────────
          case 'response.audio.done':
            browserWs.send(JSON.stringify({ type: 'response_audio_done' }));
            break;

          // ── Full response complete ───────────────────────────
          case 'response.done':
            browserWs.send(JSON.stringify({
              type:          'response_done',
              questionCount: questionCount,
            }));
            break;

          // ── Alex is "thinking" (response generating) ────────
          case 'response.created':
            browserWs.send(JSON.stringify({ type: 'response_started' }));
            break;

          // ── Error from OpenAI ────────────────────────────────
          case 'error':
            console.error('OpenAI error:', event.error);
            browserWs.send(JSON.stringify({
              type:    'error',
              message: event.error?.message || 'OpenAI Realtime error',
            }));
            break;

          // ── Ignore audio transcript deltas (we use completed) ─
          case 'conversation.item.input_audio_transcription.delta':
            break;

          default:
            // Forward all other events to browser for debugging
            // Remove this in production
            break;
        }
      });

      openaiWs.on('error', (err) => {
        console.error('OpenAI WS error:', err.message);
        browserWs.send(JSON.stringify({
          type:    'error',
          message: `Connection error: ${err.message}`,
        }));
      });

      openaiWs.on('close', (code, reason) => {
        console.log('OpenAI WS closed:', code, reason.toString());
        browserWs.send(JSON.stringify({ type: 'disconnected' }));
      });

      return;
    }

    // ── Manual commit (user pressed stop button) ──────────────
    if (msg.type === 'commit_audio') {
      if (openaiWs?.readyState === WebSocket.OPEN) {
        openaiWs.send(JSON.stringify({
          type: 'input_audio_buffer.commit',
        }));
      }
      return;
    }

    // ── Trigger Alex's response manually (if needed) ──────────
    if (msg.type === 'create_response') {
      if (openaiWs?.readyState === WebSocket.OPEN) {
        openaiWs.send(JSON.stringify({ type: 'response.create' }));
      }
      return;
    }

    // ── Update session (e.g. question count changed) ──────────
    if (msg.type === 'update_session') {
      if (openaiWs?.readyState === WebSocket.OPEN) {
        questionCount = msg.questionCount || questionCount;
        openaiWs.send(
          JSON.stringify({
            type: "session.update",
            session: {
              instructions: buildInterviewerPrompt(
                resumeText,
                skills,
                questionCount,
              ),
            },
          }),
        );
        // 🔥 Trigger first question from Alex
        openaiWs.send(
          JSON.stringify({
            type: "response.create",
            response: {
              modalities: ["audio", "text"],
              instructions:
                "Start the interview by asking the first question based on the resume.",
            },
          }),
        );
      }
      return;
    }
  });

  // ── Browser disconnected ──────────────────────────────────────
  browserWs.on('close', () => {
    console.log('🔌 Browser disconnected');
    if (openaiWs?.readyState === WebSocket.OPEN) {
      openaiWs.close();
    }
  });

  browserWs.on('error', (err) => {
    console.error('Browser WS error:', err.message);
  });
});

// ─────────────────────────────────────────────────────────────────
// INTERVIEWER SYSTEM PROMPT
// Sent to OpenAI on session.update.
// This controls Alex's entire personality and behavior.
// ─────────────────────────────────────────────────────────────────

function buildInterviewerPrompt(resumeText, skills, questionCount) {
  const progressNote = questionCount > 0
    ? `\nYou have asked ${questionCount} questions so far.${questionCount >= 7 ? ' You have 1-2 questions remaining.' : ''}`
    : '';

  return `You are Alex, a Senior Technical Interviewer at a top tech company conducting a real job interview via voice.

VOICE INTERVIEW RULES (critical):
- Keep ALL responses under 3 sentences. You are speaking out loud, not writing.
- Never use bullet points, numbered lists, markdown, or asterisks. Speak naturally.
- Ask exactly ONE question per turn. Never stack two questions.
- Do not say "Great!" or "Awesome!" — these sound fake. Use: "I see.", "Right.", "Interesting." or just proceed.
- Speak like a real human interviewer, not a chatbot.

CANDIDATE RESUME:
${resumeText.slice(0, 1500)}
Key skills: ${skills || 'Software engineering'}
${progressNote}

INTERVIEW STRUCTURE:
- Turn 1: Warm greeting + ask for self-introduction
- Turns 2-4: Technical questions based on their skills
- Turns 5-6: Behavioral questions (STAR format)  
- Turns 7-8: Situational / problem-solving
- Turn 9+: Wrap up naturally

ADAPTIVE BEHAVIOR:
- If answer was hesitant or vague: "Can you walk me through a specific example of that?"
- If answer was strong: increase difficulty next question
- If answer was very short: "Tell me more about that."
- If candidate seems stressed: "Take your time, there's no rush."

NEVER: reveal scores, say you're an AI, break character, use filler phrases like "Certainly!" or "Of course!".`;
}