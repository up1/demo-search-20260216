using Npgsql;
using Ollama;
using Pgvector;
using Pgvector.Npgsql;

// --- Configuration ---
const string connectionString = "Host=localhost;Port=5432;Database=mydb;Username=user;Password=password";
var ollamaBaseUri = new Uri("http://152.42.202.40:11434/api");
const string embeddingModel = "bge-m3";

// --- Setup Npgsql with pgvector support ---
var dataSourceBuilder = new NpgsqlDataSourceBuilder(connectionString);
dataSourceBuilder.UseVector();
await using var dataSource = dataSourceBuilder.Build();
await using var conn = await dataSource.OpenConnectionAsync();

// Ensure vector extension is enabled
await using (var cmd = new NpgsqlCommand("CREATE EXTENSION IF NOT EXISTS vector", conn))
    await cmd.ExecuteNonQueryAsync();

conn.ReloadTypes();

// --- Setup Ollama client ---
using var ollama = new OllamaApiClient(baseUri: ollamaBaseUri);

// --- Step 1: Read all documents ---
Console.WriteLine("Reading documents from database...");
var documents = new List<(int Id, string SearchText)>();

await using (var cmd = new NpgsqlCommand("SELECT id, search_text FROM documents", conn))
await using (var reader = await cmd.ExecuteReaderAsync())
{
    while (await reader.ReadAsync())
    {
        var id = reader.GetInt32(0);
        var searchText = reader.GetString(1);
        documents.Add((id, searchText));
    }
}

Console.WriteLine($"Found {documents.Count} documents to process.");

// --- Step 2 & 3: Generate embeddings and update ---
int processed = 0;
foreach (var (id, searchText) in documents)
{
    try
    {
        // Generate embedding via Ollama
        var response = await ollama.Embeddings.GenerateEmbeddingAsync(
            model: embeddingModel,
            prompt: searchText);

        if (response.Embedding == null || response.Embedding.Count == 0)
        {
            Console.WriteLine($"[SKIP] ID={id}: No embedding returned.");
            continue;
        }

        // Convert to pgvector Vector
        var vector = new Vector(response.Embedding.Select(x => (float)x).ToArray());

        // Update the embedding column
        await using var updateCmd = new NpgsqlCommand(
            "UPDATE documents SET embedding = $1 WHERE id = $2", conn);
        updateCmd.Parameters.AddWithValue(vector);
        updateCmd.Parameters.AddWithValue(id);
        await updateCmd.ExecuteNonQueryAsync();

        processed++;
        Console.WriteLine($"[OK] ID={id} - Embedded ({vector.ToArray().Length} dims) - {processed}/{documents.Count}");
    }
    catch (Exception ex)
    {
        Console.WriteLine($"[ERROR] ID={id}: {ex.Message}");
    }
}

Console.WriteLine($"\nDone! Processed {processed}/{documents.Count} documents.");
