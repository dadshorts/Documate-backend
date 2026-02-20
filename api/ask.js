import Anthropic from "@anthropic-ai/sdk";
import { Pinecone } from "@pinecone-database/pinecone";

const anthropic = new Anthropic({ apiKey: process.env.ANTHROPIC_API_KEY });
const pc = new Pinecone({ apiKey: process.env.PINECONE_API_KEY });
const index = pc.Index("documate-index");

async function embedQuestion(question) {
  const response = await fetch("https://api.openai.com/v1/embeddings", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${process.env.OPENAI_API_KEY}`,
    },
    body: JSON.stringify({
      model: "text-embedding-ada-002",
      input: question,
    }),
  });
  const data = await response.json();
  return data.data[0].embedding;
}

async function getRelevantChunks(question) {
  const vector = await embedQuestion(question);
  const results = await index.query({
    vector,
    topK: 5,
    includeMetadata: true,
  });
  return results.matches.map((m) => m.metadata);
}

export default async function handler(req, res) {
  res.setHeader("Access-Control-Allow-Origin", "*");
  res.setHeader("Access-Control-Allow-Methods", "POST, OPTIONS");
  res.setHeader("Access-Control-Allow-Headers", "Content-Type");

  if (req.method === "OPTIONS") return res.status(200).end();
  if (req.method !== "POST")
    return res.status(405).json({ error: "Method not allowed" });

  const { question } = req.body;
  if (!question) return res.status(400).json({ error: "Question is required" });

  try {
    // 1. Search Pinecone for relevant chunks
    const chunks = await getRelevantChunks(question);
    const context = chunks.map((c) => c.text || "").join("\n\n---\n\n");

    // 2. Build prompt with context
    const prompt = context
      ? `You are a ServiceNow expert. Use the following documentation to answer the question. If the answer isn't in the docs, say so and answer from your general knowledge.

DOCUMENTATION CONTEXT:
${context}

QUESTION: ${question}`
      : `You are a ServiceNow expert. Answer this question: ${question}`;

    // 3. Call Claude
    const message = await anthropic.messages.create({
      model: "claude-sonnet-4-20250514",
      max_tokens: 1024,
      messages: [{ role: "user", content: prompt }],
    });

    res.status(200).json({ answer: message.content[0].text });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
}
