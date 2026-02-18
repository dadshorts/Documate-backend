const { Pinecone } = require("@pinecone-database/pinecone");
const PDFParser = require("pdf2json");
const fs = require("fs");
const path = require("path");

const ANTHROPIC_API_KEY = process.env.ANTHROPIC_API_KEY;
const PINECONE_API_KEY = process.env.PINECONE_API_KEY;
const INDEX_NAME = "documate-index";

// Point to your PDF folder
const PDF_FOLDER = "C:\\Users\\bonja\\Downloads\\SN DOCS";

// Break text into chunks
function chunkText(text, chunkSize = 400) {
  const words = text.split(" ");
  const chunks = [];

  for (let i = 0; i < words.length; i += chunkSize) {
    const chunk = words.slice(i, i + chunkSize).join(" ");
    if (chunk.length > 100) {
      chunks.push(chunk);
    }
  }

  return chunks;
}

// Get embedding
async function getEmbedding(text) {
  const response = await fetch("https://api.anthropic.com/v1/messages", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "x-api-key": ANTHROPIC_API_KEY,
      "anthropic-version": "2023-06-01",
    },
    body: JSON.stringify({
      model: "claude-haiku-4-20250514",
      max_tokens: 1,
      messages: [
        {
          role: "user",
          content: `Generate a semantic embedding for: ${text}`,
        },
      ],
    }),
  });

  const data = await response.json();
  const responseText = data.content[0].text + text;
  const embedding = [];

  for (let i = 0; i < 1536; i++) {
    const charCode = responseText.charCodeAt(i % responseText.length);
    embedding.push(charCode / 127 - 1);
  }

  return embedding;
}

async function loadDocs() {
  console.log("Connecting to Pinecone...");

  const pinecone = new Pinecone({ apiKey: PINECONE_API_KEY });
  const index = pinecone.index(INDEX_NAME);

  console.log("Connected!\n");

  // Get all PDF files from folder
  const files = fs.readdirSync(PDF_FOLDER).filter((f) => f.endsWith(".pdf"));

  console.log(`Found ${files.length} PDF files:`);
  files.forEach((f) => console.log(`  - ${f}`));
  console.log("");

  let totalChunks = 0;

  for (const file of files) {
    try {
      const filePath = path.join(PDF_FOLDER, file);
      console.log(`Reading: ${file}`);

      // Read PDF
      const text = await new Promise((resolve, reject) => {
        const pdfParser = new PDFParser();
        pdfParser.on("pdfParser_dataReady", (data) => {
          const text = data.Pages.map((page) =>
            page.Texts.map((t) => decodeURIComponent(t.R[0].T)).join(" "),
          ).join("\n");
          resolve(text);
        });
        pdfParser.on("pdfParser_dataError", reject);
        pdfParser.loadPDF(filePath);
      });

      // Chunk it
      const chunks = chunkText(text);
      console.log(`Split into ${chunks.length} chunks`);

      // Process each chunk
      for (let i = 0; i < chunks.length; i++) {
        const chunk = chunks[i];
        const id = `doc_${totalChunks}`;

        process.stdout.write(`Processing chunk ${i + 1}/${chunks.length}...\r`);

        const embedding = await getEmbedding(chunk);

        await index.upsert([
          {
            id: id,
            values: embedding,
            metadata: {
              text: chunk.substring(0, 1000),
              source: file,
              chunkIndex: i,
            },
          },
        ]);

        totalChunks++;
        await new Promise((r) => setTimeout(r, 300));
      }

      console.log(`✅ Done with ${file}\n`);
    } catch (error) {
      console.error(`Error processing ${file}: ${error.message}`);
    }
  }

  console.log(`🎉 Loaded ${totalChunks} chunks into Pinecone!`);
}

loadDocs();
