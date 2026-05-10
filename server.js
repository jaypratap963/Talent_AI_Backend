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
const SIMLI_KEY = process.env.SIMLI_API_KEY;

if (!OPENAI_KEY) {
  console.error("❌ OPENAI_API_KEY missing from .env");
  process.exit(1);
}

import http from "http";

// Create HTTP server that handles both WebSocket and REST
const server = http.createServer(async (req, res) => {
  // CORS headers so Netlify frontend can call this
  res.setHeader("Access-Control-Allow-Origin", "*");
  res.setHeader("Access-Control-Allow-Headers", "Content-Type");

  if (req.method === "OPTIONS") {
    res.writeHead(204);
    res.end();
    return;
  }

  if (req.method === "POST" && req.url === "/evaluate") {
    let body = "";
    req.on("data", (chunk) => (body += chunk));
    req.on("end", async () => {
      try {
        const { chatHistory, resumeText } = JSON.parse(body);

        const userMessages = chatHistory.filter((m) => m.role === "user");
        const assistantMessages = chatHistory.filter(
          (m) => m.role === "assistant",
        );

        const prompt = `You are a strict, senior hiring manager evaluating a job interview. Be honest and direct — do not inflate scores to spare feelings. Evaluate ONLY what the candidate actually said. If they gave short or vague answers, reflect that with genuinely low scores.

CANDIDATE RESUME:
${resumeText}

INTERVIEW TRANSCRIPT (${userMessages.length} candidate responses, ${assistantMessages.length} interviewer turns):
${chatHistory.map((m) => `[${m.role === "user" ? "CANDIDATE" : "INTERVIEWER"}]: ${m.content}`).join("\n\n")}

═══════════════════════════════════════════════
SCORING RULES — apply strictly:

overallScore (0–100):
- 85–100: Exceptional — specific, structured, impressive answers with real examples and metrics
- 70–84:  Good — solid answers with examples, only minor gaps
- 50–69:  Average — answered but lacked depth, specifics, or structure
- 30–49:  Below average — vague, short, or struggled noticeably
- 0–29:   Poor — mostly 1–2 sentence answers, off-topic, or barely engaged
→ If fewer than 3 substantive responses: cap overallScore at 30.
→ Each answer under 20 words: treat it as a significant red flag.

technicalScore (0–100):
- If technical questions were asked: score on accuracy, depth, and real examples
- If the candidate said "I don't know" or deflected: penalise heavily
- If no technical questions were asked: match overallScore

communicationScore (0–100):
- Score on clarity, structure, and ease of following
- 1–2 sentence answers: max 40. Rambling without structure: max 55.

confidenceScore (0–100):
- Penalise: "I don't know", "I think", "maybe", "not sure", very short answers
- Reward: direct, assertive, example-backed answers

═══════════════════════════════════════════════
STRENGTHS — format requirements:
- List only genuine strengths backed by something actually said
- Be specific: "Explained how they used OpenAI to auto-generate exam questions at GLA University" not "Good technical knowledge"
- If there are no real strengths, say so honestly (e.g., "Answers were too brief to identify clear strengths")
- Exactly 3 items

IMPROVEMENTS — format requirements (MOST IMPORTANT):
Each improvement MUST follow this format:
  "[What they did] → Instead, you could say: '[Specific stronger answer using their resume]'"

Rules:
- Directly reference what was said (or not said) in the transcript
- Suggest a better answer that uses SPECIFIC items from their resume (projects, companies, metrics, technologies)
- If they gave a vague answer about a project, suggest they mention the actual system from their resume with a concrete outcome
- Include at least one tip for redirecting when struggling (e.g., "If unsure, pivot: 'In my examination platform at GLA, I solved a similar challenge by...'")
- Exactly 3 items

SUMMARY — 2–3 sentences:
- Name specific things said or not said
- Reference actual resume items they failed to leverage
- Be direct about the overall performance level

═══════════════════════════════════════════════
EXAMPLE of a good improvements item:
"Only said 'I worked on Angular apps' without any detail → Instead say: 'At GLA University I built an Angular-based examination platform handling hundreds of concurrent users — I can walk you through how I optimised the API layer for throughput.'"

Respond with ONLY valid JSON, no markdown:
{
  "overallScore": <0-100>,
  "technicalScore": <0-100>,
  "communicationScore": <0-100>,
  "confidenceScore": <0-100>,
  "strengths": ["...", "...", "..."],
  "improvements": ["...", "...", "..."],
  "summary": "..."
}`;

        const openaiRes = await fetch(
          "https://api.openai.com/v1/chat/completions",
          {
            method: "POST",
            headers: {
              "Content-Type": "application/json",
              Authorization: `Bearer ${OPENAI_KEY}`,
            },
            body: JSON.stringify({
              model: "gpt-4o-mini",
              temperature: 0.3,
              messages: [{ role: "user", content: prompt }],
            }),
          },
        );

        const data = await openaiRes.json();
        const text = data.choices?.[0]?.message?.content ?? "";
        const clean = text.replace(/```json|```/g, "").trim();
        const result = JSON.parse(clean);

        res.writeHead(200, { "Content-Type": "application/json" });
        res.end(JSON.stringify(result));
      } catch (err) {
        console.error("Evaluation error:", err);
        res.writeHead(500);
        res.end(JSON.stringify({ error: err.message }));
      }
    });
    return;
  }

  if (req.method === "POST" && req.url === "/annotate") {
    let body = "";
    req.on("data", (chunk) => (body += chunk));
    req.on("end", async () => {
      try {
        const { turns, resumeText } = JSON.parse(body);

        const prompt = `You are a senior interview coach. For each candidate answer below, provide specific annotation.

Resume context: ${resumeText}

For each turn, respond with a JSON array. Each item must have:
- turnIndex: number (from input)
- question: string (from input)  
- answer: string (from input)
- label: one of "strong", "adequate", "weak", "missed"
  strong = detailed, specific, impressive answer
  adequate = answered but lacked depth or examples
  weak = vague, too short, or off-topic
  missed = completely avoided or misunderstood the question
- feedback: 1-2 sentences of SPECIFIC feedback referencing what they actually said
- betterAnswer: 2-3 sentences showing what a strong answer would look like for this specific question

Turns to annotate:
${turns.map((t) => `Turn ${t.turnIndex}:\nQ: ${t.question}\nA: ${t.answer}`).join("\n\n")}

Respond with ONLY a valid JSON array, no markdown, no explanation.`;

        const openaiRes = await fetch(
          "https://api.openai.com/v1/chat/completions",
          {
            method: "POST",
            headers: {
              "Content-Type": "application/json",
              Authorization: `Bearer ${OPENAI_KEY}`,
            },
            body: JSON.stringify({
              model: "gpt-4o-mini",
              temperature: 0.3,
              messages: [{ role: "user", content: prompt }],
            }),
          },
        );

        const data = await openaiRes.json();
        const text = data.choices?.[0]?.message?.content ?? "[]";
        const clean = text.replace(/```json|```/g, "").trim();
        const result = JSON.parse(clean);

        res.writeHead(200, { "Content-Type": "application/json" });
        res.end(JSON.stringify(result));
      } catch (err) {
        console.error("Annotation error:", err);
        res.writeHead(500);
        res.end(JSON.stringify({ error: err.message }));
      }
    });
    return;
  }

  if (req.method === "POST" && req.url === "/simli/start") {
    try {
      const simliRes = await fetch("https://api.simli.ai/compose/token", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "x-simli-api-key": SIMLI_KEY,
        },
        body: JSON.stringify({
          faceId: "cace3ef7-a4c4-425d-a8cf-a5358eb0c427", // note: 7c at the end
          maxSessionLength: 3600,
          maxIdleTime: 300,
          handleSilence: true,
        }),
      });
      const data = await simliRes.json();
      console.log("Simli token response:", data);
      res.writeHead(200, { "Content-Type": "application/json" });
      res.end(JSON.stringify(data));
    } catch (err) {
      console.error("Simli error:", err);
      res.writeHead(500);
      res.end(JSON.stringify({ error: err.message }));
    }
    return;
  }

  if (req.method === "GET" && req.url === "/simli/ice") {
    try {
      const iceRes = await fetch("https://api.simli.ai/compose/ice", {
        method: "GET",
        headers: { "x-simli-api-key": SIMLI_KEY },
      });
      const data = await iceRes.json();
      res.writeHead(200, { "Content-Type": "application/json" });
      res.end(JSON.stringify(data));
    } catch (err) {
      res.writeHead(500);
      res.end(JSON.stringify({ error: err.message }));
    }
    return;
  }

  res.writeHead(404);
  res.end();
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
              voice: "shimmer",

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
          console.log("OpenAI →", event.type);
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
          // Sending 'ready' there resets frontend phase while Madeline is speaking.
          case "session.created":
            sessionReady = true;
            browserWs.send(JSON.stringify({ type: "ready" }));
            // Do NOT fire response.create here.
            // Frontend sends 'start_interview' only after Simli avatar is
            // connected and rendering — so Madeline's lips sync from word one.
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

          // ── Madeline's response text streaming ──────────────────
          case "response.text.delta":
            browserWs.send(
              JSON.stringify({
                type: "response_text_delta",
                delta: event.delta,
              }),
            );
            break;

          // ── Madeline's response text complete ───────────────────
          case "response.text.done":
            break;

          // ── Madeline's audio streaming (PCM16 chunks) ───────────
          // This is the hot path — fires many times per second
          case "response.audio.delta":
            browserWs.send(
              JSON.stringify({
                type: "response_audio_delta",
                delta: event.delta, // base64 PCM16 chunk
              }),
            );
            break;

          // ── Madeline's audio complete ────────────────────────────
          case "response.audio.done":
            browserWs.send(JSON.stringify({ type: "response_audio_done" }));
            break;

          // REPLACE the response.done case with this:
          // ── Extract Madeline's text reliably from response.done ──────
          case "response.done": {
            try {
              const output = event.response?.output ?? [];
              for (const item of output) {
                let fullText = "";

                for (const part of item.content ?? []) {
                  if (part.type === "text" && part.text?.trim()) {
                    // Text modality output (if it exists)
                    fullText = part.text.trim();
                  } else if (part.type === "audio" && part.transcript?.trim()) {
                    // Audio transcript — THIS is what actually fires
                    fullText = fullText || part.transcript.trim();
                  }
                }

                if (fullText) {
                  const isComplete = fullText.includes("INTERVIEW_COMPLETE");
                  const cleanText = fullText
                    .replace("INTERVIEW_COMPLETE", "")
                    .trim();
                  questionCount++;
                  console.log(
                    `📤 Madeline [Q${questionCount}]:`,
                    cleanText.slice(0, 60),
                  );
                  browserWs.send(
                    JSON.stringify({
                      type: "response_text_done",
                      text: cleanText,
                      interviewDone: isComplete,
                      questionCount,
                    }),
                  );
                }
              }
            } catch (e) {
              console.error("response.done text extraction failed:", e);
            }

            browserWs.send(
              JSON.stringify({ type: "response_done", questionCount }),
            );
            break;
          }

          // ── Madeline is "thinking" (response generating) ────────
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

    // ── Start interview — sent by frontend after Simli is ready ─
    // Guarantees avatar is rendering before Madeline speaks first word.
    // REPLACE WITH:
    if (msg.type === "start_interview") {
      if (openaiWs?.readyState === WebSocket.OPEN) {
        // Truncate any conversation items that might exist from a previous session
        // that wasn't fully closed. This guarantees Maya starts with zero memory.
        openaiWs.send(
          JSON.stringify({
            type: "conversation.item.truncate",
            item_id: "root",
            content_index: 0,
            audio_end_ms: 0,
          }),
        );

        // Small delay to let truncate settle, then trigger first question
        setTimeout(() => {
          if (openaiWs?.readyState === WebSocket.OPEN) {
            openaiWs.send(
              JSON.stringify({
                type: "response.create",
                response: {
                  modalities: ["audio", "text"],
                  instructions:
                    "Start the interview now. Greet the candidate warmly and ask them to introduce themselves. Keep it to 2-3 natural sentences.",
                },
              }),
            );
          }
        }, 100);
      }
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

    // ── Trigger Madeline's response manually (if needed) ──────────
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
        // audio and starts a new response — cutting Madeline off mid-sentence.
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
// This controls Madeline's entire personality and behavior.
// ─────────────────────────────────────────────────────────────────

function buildInterviewerPrompt(resumeText, skills, questionCount) {
  const isTechnical =
    /engineer|developer|software|coding|programming|data|devops|architect/i.test(
      resumeText,
    );

  const progressNote =
    questionCount >= 10
      ? `\nYou have asked ${questionCount} questions. Start wrapping up — ask one final question then close the interview.`
      : questionCount >= 7
        ? `\nYou have asked ${questionCount} questions. You may continue if the candidate is engaged and answers are rich. Otherwise begin wrapping up.`
        : questionCount > 0
          ? `\nYou have asked ${questionCount} questions so far.`
          : "";

  // Build a randomised question pool so every interview feels different
  const technicalPool = [
    "Ask them to walk through how they would design a system they've never built before — pick something relevant from their resume.",
    "Give a real scenario: 'A critical bug hits production at 2am. Walk me through exactly what you do.'",
    "Ask a 'why did you choose X over Y' question about a specific technology on their resume.",
    "Ask them to explain the hardest technical problem they've ever debugged — what made it hard and how did they find it.",
    "Ask: 'Walk me through a time your code broke something in production. What happened and what did you change after?'",
    "Ask a concrete architecture question: 'If you had to scale your current project to 10x the users, what breaks first?'",
    "Ask about testing: 'How do you decide what to test and what not to test? Give me a real example from your work.'",
    "Pick one skill from their resume and ask: 'Give me the most advanced thing you've done with [skill]. Walk me through it.'",
  ];

  const behavioralPool = [
    "Ask: 'Tell me about a time you strongly disagreed with a technical decision made by your team. What did you do?'",
    "Ask: 'Describe a project that failed or went badly. What was your role and what did you learn?'",
    "Ask: 'Tell me about the most difficult colleague you've worked with. How did you handle it?'",
    "Ask: 'Walk me through a time you had to learn something completely new under a tight deadline.'",
    "Ask: 'Tell me about a time you pushed back on a requirement from a stakeholder or manager. How did it go?'",
    "Ask: 'Describe a time you made a decision without enough information. How did you handle the uncertainty?'",
    "Ask: 'Tell me about the project you are most proud of. Not because it went well — because of what you personally contributed.'",
  ];

  const careerPool = [
    "Ask: 'Why are you looking to leave your current role? What specifically is missing for you there?'",
    "Ask: 'Looking at your resume, I notice [observe something — a gap, a short tenure, a career change]. Can you walk me through that?'",
    "Ask: 'What does your ideal next role look like — not the title, but the actual day-to-day work?'",
    "Ask: 'Where do you see yourself in 3 years? I'm not looking for a rehearsed answer — what do you actually want to build or become?'",
    "Ask: 'What is the one skill or area you are most actively working to improve right now?'",
    "Ask about hobbies or life outside work: 'What do you do outside of work that you are genuinely passionate about? Does any of it connect to how you think or work?'",
  ];

  const closingPool = [
    "Ask: 'Is there anything on your resume that you feel we haven't covered that you'd really like me to know about?'",
    "Ask: 'What question were you hoping I would ask today that I haven't asked?'",
    "Ask: 'Do you have any questions for me about the role or the team?'",
  ];

  // Pick randomly from each pool so questions vary across interviews
  const pick = (arr) => arr[Math.floor(Math.random() * arr.length)];
  const chosenTech1 = pick(technicalPool);
  const chosenTech2 = pick(technicalPool.filter((q) => q !== chosenTech1));
  const chosenBeh1 = pick(behavioralPool);
  const chosenBeh2 = pick(behavioralPool.filter((q) => q !== chosenBeh1));
  const chosenCareer = pick(careerPool);
  const chosenClose = pick(closingPool);

  return `You are Madeline, a sharp and experienced senior interviewer conducting a real job interview over a voice call.

YOUR PERSONA:
- Direct, warm, and genuinely curious — not robotic, not overly formal
- You listen carefully and follow up on what the candidate says — not just the next scripted question
- You adapt your depth based on how the candidate answers: strong answers get harder follow-ups; weak answers get gentle probes for more
- You sound like a real senior professional, not a chatbot

VOICE RULES — follow these strictly:
- Speak in 2–4 natural sentences per turn. Not too short (sounds cold), not too long (hard to follow by ear)
- Never use bullet points, numbers, markdown, asterisks, or lists — this is spoken audio
- Ask ONE question per turn only — never stack two questions
- Avoid hollow affirmations: never say "Great!", "Awesome!", "Certainly!", "Of course!", "Sure!"
- Use natural acknowledgements: "I see.", "That makes sense.", "Interesting.", "Right.", "Go on."

CANDIDATE RESUME:
${resumeText.slice(0, 2000)}
Key skills detected: ${skills || "as mentioned in the resume"}
${
  isTechnical
    ? "This is a technical candidate. Probe technical depth, architecture thinking, and engineering judgement."
    : "This candidate may not be in a technical role. Focus on domain expertise, stakeholder management, and judgement calls."
}
${progressNote}

INTERVIEW STRUCTURE — follow this flow but stay adaptive:
- Turn 1: Warm greeting. Ask them to briefly introduce themselves and what they are currently working on.
- Turn 2: ${chosenTech1}
- Turn 3: Follow up naturally on their answer OR ${chosenBeh1}
- Turn 4: ${chosenCareer}
- Turn 5: ${chosenTech2}
- Turn 6: ${chosenBeh2}
- Turn 7 onwards: Continue naturally. You may ask ANY of these if not yet covered:
    * Ask about a gap or short tenure you notice on the resume
    * Ask about their hobbies or life outside work
    * Ask a scenario question relevant to their domain
    * ${chosenClose}
  Continue as long as the conversation is flowing well. You can go beyond 8 turns if the candidate is engaged and giving rich answers. Aim for depth, not a checklist.

ADAPTIVE BEHAVIOUR:
- Vague or weak answer → "Can you give me a specific example of that?"
- Strong, detailed answer → raise the bar: ask a harder or more nuanced follow-up
- Very short answer → "Tell me a bit more about that."
- Candidate seems nervous → "Take your time, there's no rush."
- Candidate gives a textbook answer → "That's the standard answer — what actually happened in your case?"

ENDING THE INTERVIEW:
- When the conversation feels complete (typically after 8–15 turns), ask the closing question you chose: "${chosenClose}"
- Once the candidate indicates they have nothing more to add, give a warm, personal closing. Thank them by name if you know it. Say the team will be in touch. Then say exactly this phrase to signal the end: INTERVIEW_COMPLETE
- Never end abruptly. Always close with warmth before INTERVIEW_COMPLETE.

NEVER: say you are an AI, reveal scoring, repeat a question you already asked, break character.`;
}
