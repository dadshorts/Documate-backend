import { Pinecone } from "@pinecone-database/pinecone";

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

export default async function handler(req, res) {
  res.setHeader("Access-Control-Allow-Origin", "*");

  const question = req.query.q || "bulk update transfer order lines";
  const vector = await embedQuestion(question);
  const results = await index.query({
    vector,
    topK: 10,
    includeMetadata: true,
  });

  const simplified = results.matches.map((m) => ({
    score: m.score.toFixed(4),
    source: m.metadata.source || "?",
    heading: m.metadata.heading || "?",
    page: m.metadata.page || "?",
    preview: (m.metadata.text || "").substring(0, 200),
  }));

  res.status(200).json(simplified);
}
