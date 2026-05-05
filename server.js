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

import { WebSocketServer, WebSocket } from "ws";
import dotenv from "dotenv";

dotenv.config();

const PORT = process.env.PORT || 3001;
const OPENAI_KEY = process.env.OPENAI_API_KEY;

if (!OPENAI_KEY) {
  console.error("❌ OPENAI_API_KEY missing from .env");
  process.exit(1);
}

import http from 'http';

// Create HTTP server that handles both WebSocket and REST
const server = http.createServer(async (req, res) => {
  // CORS headers so Netlify frontend can call this
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');

  if (req.method === 'OPTIONS') {
    res.writeHead(204); res.end(); return;
  }

  if (req.method === 'POST' && req.url === '/evaluate') {
    let body = '';
    req.on('data', chunk => body += chunk);
    req.on('end', async () => {
      try {
        const { chatHistory, resumeText } = JSON.parse(body);

        const prompt = `You are evaluating a job interview. Based on the conversation below, provide scores and feedback.

Resume summary: ${resumeText}

Interview conversation:
${chatHistory.map(m => `${m.role.toUpperCase()}: ${m.content}`).join('\n')}

Respond with ONLY valid JSON in exactly this format:
{
  "overallScore": <0-100>,
  "technicalScore": <0-100>,
  "communicationScore": <0-100>,
  "confidenceScore": <0-100>,
  "strengths": ["...", "...", "..."],
  "improvements": ["...", "...", "..."],
  "summary": "2-3 sentence summary of the candidate's performance"
}`;

        const openaiRes = await fetch('https://api.openai.com/v1/chat/completions', {
          method: 'POST',
          headers: {
            'Content-Type':  'application/json',
            'Authorization': `Bearer ${OPENAI_KEY}`,
          },
          body: JSON.stringify({
            model:       'gpt-4o-mini',
            temperature: 0.3,
            messages: [{ role: 'user', content: prompt }],
          }),
        });

        const data     = await openaiRes.json();
        const text     = data.choices?.[0]?.message?.content ?? '';
        const clean    = text.replace(/```json|```/g, '').trim();
        const result   = JSON.parse(clean);

        res.writeHead(200, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify(result));

      } catch (err) {
        console.error('Evaluation error:', err);
        res.writeHead(500);
        res.end(JSON.stringify({ error: err.message }));
      }
    });
    return;
  }

  res.writeHead(404); res.end();
});

// Attach WebSocket server to the same HTTP server
const wss = new WebSocketServer({ server });
server.listen(PORT, () => {
  console.log(`✅ TalentAI server running on port ${PORT}`);
});

// ── Per-connection handler ────────────────────────────────────────
wss.on("connection", (browserWs) => {
  console.log("🔌 Browser client connected");

  let openaiWs = null; // WebSocket to OpenAI Realtime
  let sessionReady = false; // true after session.created received
  let resumeText = "";
  let skills = "";
  let questionCount = 0; // track questions for prompt adaptation

  // ── Handle messages from Angular browser ─────────────────────
  browserWs.on("message", (data, isBinary) => {
    // Binary = raw PCM16 audio chunk from mic
    if (isBinary) {
      if (openaiWs?.readyState === WebSocket.OPEN) {
        // Forward raw audio to OpenAI input buffer
        const b64 = Buffer.from(data).toString("base64");
        openaiWs.send(
          JSON.stringify({
            type: "input_audio_buffer.append",
            audio: b64,
          }),
        );
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
    if (msg.type === "init") {
      resumeText = msg.resumeText || "";
      skills = msg.skills || "";
      questionCount = msg.questionCount || 0;

      // Open connection to OpenAI Realtime API
      openaiWs = new WebSocket(
        "wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview-2024-12-17",
        {
          headers: {
            Authorization: `Bearer ${OPENAI_KEY}`,
            "OpenAI-Beta": "realtime=v1",
          },
        },
      );

      openaiWs.on("open", () => {
        console.log("✅ Connected to OpenAI Realtime");

        // ── Configure the session ─────────────────────────────
        // This is the most important message — sets VAD, voice,
        // instructions, and audio format
        openaiWs.send(
          JSON.stringify({
            type: "session.update",
            session: {
              // ── Modalities: audio + text ──────────────────────
              modalities: ["audio", "text"],

              // ── System instructions (interviewer persona) ─────
              instructions: buildInterviewerPrompt(
                resumeText,
                skills,
                questionCount,
              ),

              // ── Voice options (uncomment the one you want) ────
              // 'ash'   = Indian male accent (default)
              // 'shimmer' = Indian female accent
              // 'echo'  = US male
              // 'alloy' = US female
              voice: "ash",

              // ── Audio format ──────────────────────────────────
              // pcm16 at 24kHz for both input and output
              // Browser must send PCM16 @ 24kHz (we handle this in frontend)
              input_audio_format: "pcm16",
              output_audio_format: "pcm16",

              // ── Transcription model ───────────────────────────
              // whisper-1 gives accurate transcripts including fillers
              input_audio_transcription: {
                model: "whisper-1",
              },

              // ── SERVER-SIDE VAD CONFIGURATION ─────────────────
              // This is the entire reason we're using Realtime API.
              // OpenAI's acoustic model detects speech boundaries.
              // No volume thresholds. No timers. Neural detection.
              turn_detection: {
                type: "server_vad",

                // How sensitive the VAD is (0.0–1.0)
                // 0.5 is default. Lower = more sensitive (picks up soft voice)
                // Higher = less sensitive (ignores background noise)
                threshold: 0.6,

                // How much audio before speech counts (ms)
                // Prevents single-word fragments from triggering
                prefix_padding_ms: 500,

                // How long silence AFTER speech before committing (ms)
                // 800ms = natural conversational pause without cutting off
                // This is tuned for interview answers (slightly longer than chat)
                silence_duration_ms: 1200,
              },

              // ── Response config ───────────────────────────────
              // Temperature 0.7 = professional but not robotic
              // Max output tokens capped at 150 = concise questions
              temperature: 0.7,
            },
          }),
        );
      });

      // ── Handle events from OpenAI → forward to browser ───────
      openaiWs.on("message", (rawEvent) => {
        let event;
        try {
          event = JSON.parse(rawEvent.toString());
        } catch {
          return;
        }

        // Log key events (remove in production)
        if (!["response.audio.delta"].includes(event.type)) {
          console.log("OpenAI →", event.type);
        }

        switch (event.type) {
          // ── Session ready ───────────────────────────────────
          // ONLY session.created sends 'ready'.
          // session.updated fires on every session.update call mid-interview.
          // Sending 'ready' there resets frontend phase while Alex is speaking.
          case "session.created":
            sessionReady = true;
            browserWs.send(JSON.stringify({ type: "ready" }));
            // Trigger Alex's opening greeting immediately after session is ready.
            // Without this, server_vad waits for the user to speak first.
            openaiWs.send(
              JSON.stringify({
                type: "response.create",
                response: {
                  modalities: ["audio", "text"],
                  instructions:
                    "Start the interview now. Greet the candidate warmly and ask them to introduce themselves.",
                },
              }),
            );
            break;

          case "session.updated":
            break; // intentionally silent

          // ── VAD: user started speaking ──────────────────────
          case "input_audio_buffer.speech_started":
            browserWs.send(JSON.stringify({ type: "user_speech_start" }));
            break;

          // ── VAD: user stopped speaking ──────────────────────
          case "input_audio_buffer.speech_stopped":
            browserWs.send(JSON.stringify({ type: "user_speech_stop" }));
            break;

          // ── VAD committed the audio for transcription ───────
          case "input_audio_buffer.committed":
            browserWs.send(JSON.stringify({ type: "user_audio_committed" }));
            break;

          // ── Live transcript of what user said ───────────────
          case "conversation.item.input_audio_transcription.completed":
            browserWs.send(
              JSON.stringify({
                type: "user_transcript",
                transcript: event.transcript,
                item_id: event.item_id,
              }),
            );
            break;

          // ── Alex's response text streaming ──────────────────
          case "response.text.delta":
            browserWs.send(
              JSON.stringify({
                type: "response_text_delta",
                delta: event.delta,
              }),
            );
            break;

          // ── Alex's response text complete ───────────────────
          case "response.text.done":
            questionCount++;
            const responseText = event.text || "";
            // Check if Alex signalled interview is complete
            const isComplete = responseText.includes("INTERVIEW_COMPLETE");
            // Strip the signal token before sending to browser
            const cleanText = responseText
              .replace("INTERVIEW_COMPLETE", "")
              .trim();
            browserWs.send(
              JSON.stringify({
                type: "response_text_done",
                text: cleanText,
                interviewDone: isComplete,
                questionCount: questionCount,
              }),
            );
            break;

          // ── Alex's audio streaming (PCM16 chunks) ───────────
          // This is the hot path — fires many times per second
          case "response.audio.delta":
            browserWs.send(
              JSON.stringify({
                type: "response_audio_delta",
                delta: event.delta, // base64 PCM16 chunk
              }),
            );
            break;

          // ── Alex's audio complete ────────────────────────────
          case "response.audio.done":
            browserWs.send(JSON.stringify({ type: "response_audio_done" }));
            break;

          // ── Full response complete ───────────────────────────
          case "response.done":
            browserWs.send(
              JSON.stringify({
                type: "response_done",
                questionCount: questionCount,
              }),
            );
            break;

          // ── Alex is "thinking" (response generating) ────────
          case "response.created":
            browserWs.send(JSON.stringify({ type: "response_started" }));
            break;

          // ── Error from OpenAI ────────────────────────────────
          case "error":
            console.error("OpenAI error:", event.error);
            browserWs.send(
              JSON.stringify({
                type: "error",
                message: event.error?.message || "OpenAI Realtime error",
              }),
            );
            break;

          // ── Ignore audio transcript deltas (we use completed) ─
          case "conversation.item.input_audio_transcription.delta":
            break;

          default:
            // Forward all other events to browser for debugging
            // Remove this in production
            break;
        }
      });

      openaiWs.on("error", (err) => {
        console.error("OpenAI WS error:", err.message);
        browserWs.send(
          JSON.stringify({
            type: "error",
            message: `Connection error: ${err.message}`,
          }),
        );
      });

      openaiWs.on("close", (code, reason) => {
        console.log("OpenAI WS closed:", code, reason.toString());
        browserWs.send(JSON.stringify({ type: "disconnected" }));
      });

      return;
    }

    // ── Manual commit (user pressed stop button) ──────────────
    if (msg.type === "commit_audio") {
      if (openaiWs?.readyState === WebSocket.OPEN) {
        openaiWs.send(
          JSON.stringify({
            type: "input_audio_buffer.commit",
          }),
        );
      }
      return;
    }

    // ── Trigger Alex's response manually (if needed) ──────────
    if (msg.type === "create_response") {
      if (openaiWs?.readyState === WebSocket.OPEN) {
        openaiWs.send(JSON.stringify({ type: "response.create" }));
      }
      return;
    }

    // ── Update session (e.g. question count changed) ──────────
    if (msg.type === "update_session") {
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
        // Do NOT send response.create here.
        // update_session fires after every answer (to update question count).
        // Sending response.create mid-interview cancels OpenAI's current
        // audio and starts a new response — cutting Alex off mid-sentence.
      }
      return;
    }
  });

  // ── Browser disconnected ──────────────────────────────────────
  browserWs.on("close", () => {
    console.log("🔌 Browser disconnected");
    if (openaiWs?.readyState === WebSocket.OPEN) {
      openaiWs.close();
    }
  });

  browserWs.on("error", (err) => {
    console.error("Browser WS error:", err.message);
  });
});

// ─────────────────────────────────────────────────────────────────
// INTERVIEWER SYSTEM PROMPT
// Sent to OpenAI on session.update.
// This controls Alex's entire personality and behavior.
// ─────────────────────────────────────────────────────────────────

function buildInterviewerPrompt(resumeText, skills, questionCount) {
  const progressNote =
    questionCount > 0
      ? `\nYou have asked ${questionCount} questions so far.${questionCount >= 7 ? " This is near the end — wrap up naturally in 1-2 more questions." : ""}`
      : "";

  // Detect domain from resume so non-technical candidates get relevant questions
  const isTechnical =
    /engineer|developer|software|coding|programming|data|devops|architect/i.test(
      resumeText,
    );
  const domainHint = isTechnical
    ? "The candidate has a technical background. Include technical depth where relevant."
    : "The candidate may not have a technical background. Focus on their domain expertise, soft skills, and situational judgment instead of code or systems.";

  return `You are Alex, an experienced interviewer conducting a real job interview over voice call.

YOUR PERSONA:
- Professional, calm, and genuinely curious — not robotic or overly formal
- You adapt your style to the candidate's field: technical for engineers, strategic for managers, creative for designers
- You sound like a real person, not a chatbot

VOICE RULES — follow these strictly:
- Speak in 2 to 4 natural sentences per turn. Not too short (sounds cold), not too long (hard to follow by ear)
- Never use bullet points, numbers, markdown, asterisks, or lists — this is spoken audio
- Ask ONE question per turn only — never stack two questions in one response
- Avoid filler affirmations: never say "Great!", "Awesome!", "Certainly!", "Of course!", "Sure!"
- Use natural acknowledgements instead: "I see.", "That makes sense.", "Interesting.", "Right."

CANDIDATE RESUME:
${resumeText.slice(0, 2000)}
Key areas: ${skills || "as mentioned in the resume above"}
${domainHint}
${progressNote}

INTERVIEW FLOW:
- Turn 1: Warm, brief greeting. Ask them to introduce themselves and their background
- Turns 2-3: Questions about their core domain — what they actually do day to day
- Turns 4-5: Dig into a specific project or achievement from their resume
- Turns 6-7: Behavioral — how they handle pressure, conflict, deadlines, or collaboration
- Turn 8: One forward-looking question — where they want to grow or how they approach learning
- Turn 9+: Ask "Do you have any questions for me, or is there anything else you'd like to add?"

ENDING THE INTERVIEW — this is critical:
- After turn 8 or when the conversation feels complete, ask: "Do you have any questions for me, or anything else you'd like to add?"
- If the candidate says anything like "No", "That's all", "No thank you", "I'm good", "Nothing else" — respond with a warm closing: thank them by name if known, say the team will be in touch, wish them well, then say exactly this phrase to signal the end: "INTERVIEW_COMPLETE"
- If they have more questions, answer them naturally, then ask again until they indicate they are done
- Never end abruptly. Always close warmly before saying INTERVIEW_COMPLETE

ADAPTIVE RULES:
- Weak or vague answer → "Can you give me a specific example of that?"
- Strong detailed answer → raise the bar slightly on the next question
- Very short answer → "Tell me a bit more about that."
- Candidate seems nervous → "Take your time, there's no rush."

NEVER: say you are an AI, reveal any scoring, break character.`;
}
