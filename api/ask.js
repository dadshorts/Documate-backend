import Anthropic from "@anthropic-ai/sdk";
import { Pinecone } from "@pinecone-database/pinecone";

const anthropic = new Anthropic({ apiKey: process.env.ANTHROPIC_API_KEY });
const pc = new Pinecone({ apiKey: process.env.PINECONE_API_KEY });
const index = pc.Index("documate-index");

const SYSTEM_PROMPT = `You are DocuMate, a ServiceNow expert whose mission is to make ServiceNow easy to use for everyone — from day-one beginners to seasoned admins.

Your teaching philosophy:
- Never assume the user knows where to click. If a workflow involves navigating menus, opening records, or clicking buttons, walk through each step explicitly.
- Use numbered steps for any procedural answer. Include the exact navigation path (e.g., "Navigate to Asset > Transfer Orders > All").
- Describe what the user should SEE at each step — what the screen looks like, what fields matter, what buttons to click.
- When a workflow has a specific UI component (like a popup, a related list, a UI action button), name it and describe how to interact with it.
- If there are common mistakes or gotchas, call them out.
- After the step-by-step, include a brief summary of WHY this process works the way it does — help the user build mental models, not just follow instructions.

When using the provided documentation:
- Ground your answer in the documentation context provided. Cite specific details from the docs.
- Each chunk includes its source document, page number, and section heading. Use this to give the user precise references.
- If the documentation covers the topic but lacks step-by-step detail, fill in the procedural gaps from your ServiceNow knowledge while noting which parts come from the docs vs. your general expertise.
- If the documentation doesn't cover the topic at all, say so clearly, then answer from your general ServiceNow knowledge and label it as such.

Formatting:
- Use markdown for readability — headers, bold for UI element names, code blocks for scripts or filter expressions.
- Keep answers thorough but scannable. A beginner should be able to follow along. An expert should be able to skim to the part they need.

Confidence & Follow-up:
- After answering, silently evaluate your confidence that the provided documentation context directly and fully answers the user's question. Rate it 0-100.
- If your confidence is 70 or above, end your response normally.
- If your confidence is below 70, append a section at the end of your answer:

---
**Want a more specific answer?** [ask 1-2 clarifying questions that would help narrow down the answer, e.g., "Are you asking about bulk receiving from a transfer order, a purchase order, or an RMA?" or "Which ServiceNow version are you on — Zurich, Yokohama, or earlier?"]

Do NOT mention the confidence score or the word "confidence" to the user. Just either ask the follow-up or don't.`;

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
    topK: 10,
    includeMetadata: true,
  });

  // Filter out low-relevance chunks (similarity score below 0.65)
  const relevant = results.matches.filter((m) => m.score >= 0.65);

  // Return top 8 after filtering
  return relevant.slice(0, 8).map((m) => ({
    text: m.metadata.text || "",
    source: m.metadata.source || "Unknown",
    page: m.metadata.page || "N/A",
    heading: m.metadata.heading || "",
    score: m.score,
  }));
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

    // 2. Build context with metadata for each chunk
    const context = chunks
      .map(
        (c, i) =>
          `[Source: ${c.source} | Page: ${c.page} | Section: ${c.heading} | Relevance: ${(c.score * 100).toFixed(0)}%]\n${c.text}`,
      )
      .join("\n\n---\n\n");

    // 3. Build prompt with context
    const userMessage = context
      ? `DOCUMENTATION CONTEXT (${chunks.length} chunks retrieved):\n\n${context}\n\n---\n\nUSER QUESTION: ${question}`
      : `USER QUESTION: ${question}\n\n(No documentation context was found for this query. Answer from your general ServiceNow expertise and note that this is not sourced from the indexed docs.)`;

    // 4. Call Claude with system prompt
    const message = await anthropic.messages.create({
      model: "claude-sonnet-4-20250514",
      max_tokens: 2048,
      system: SYSTEM_PROMPT,
      messages: [{ role: "user", content: userMessage }],
    });

    res.status(200).json({ answer: message.content[0].text });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
}
