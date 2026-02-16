using Npgsql;
using Ollama;
using Pgvector;
using Pgvector.Npgsql;

// --- Parse command-line arguments ---
string? processName = null;
string? inputText = null;
for (int i = 0; i < args.Length; i++)
{
    if (args[i] == "--process" && i + 1 < args.Length)
        processName = args[++i];
    else if (args[i] == "--input" && i + 1 < args.Length)
        inputText = args[++i];
}

if (string.IsNullOrEmpty(processName))
{
    Console.WriteLine("Usage: dotnet run -- --process <command>");
    Console.WriteLine("Available commands:");
    Console.WriteLine("  migrate                          Read documents, generate embeddings via Ollama, and update the database");
    Console.WriteLine("  search --input \"<query>\"         Semantic search documents by input text");
    
    return;
}

switch (processName)
{
    case "migrate":
        await RunMigrate();
        break;
    case "search":
        if (string.IsNullOrEmpty(inputText))
        {
            Console.WriteLine("Error: --input is required for search.");
            Console.WriteLine("Usage: dotnet run -- --process search --input \"your query\"");
            return;
        }
        await RunSearch(inputText);
        break;
    default:
        Console.WriteLine($"Unknown process: {processName}");
        Console.WriteLine("Available commands: migrate, search");
        // Add example commands for clarity
        Console.WriteLine("\nExample usage:");
        Console.WriteLine("  dotnet run -- --process migrate");
        Console.WriteLine("  dotnet run -- --process search --input \"เอกสาร\"");
        break;
}

async Task RunSearch(string input)
{
    // --- Configuration ---
    const string connectionString = "Host=localhost;Port=5432;Database=mydb;Username=user;Password=password";
    var ollamaBaseUri = new Uri("http://152.42.202.40:11434/api");
    const string embeddingModel = "bge-m3";

    // --- Setup Npgsql with pgvector support ---
    var dataSourceBuilder = new NpgsqlDataSourceBuilder(connectionString);
    dataSourceBuilder.UseVector();
    await using var dataSource = dataSourceBuilder.Build();
    await using var conn = await dataSource.OpenConnectionAsync();
    conn.ReloadTypes();

    // --- Setup Ollama client ---
    using var ollama = new OllamaApiClient(baseUri: ollamaBaseUri);

    // --- Step 1: Embed the user input ---
    Console.WriteLine($"Searching for: \"{input}\"");
    Console.WriteLine("Generating embedding...");

    var response = await ollama.Embeddings.GenerateEmbeddingAsync(
        model: embeddingModel,
        prompt: input);

    if (response.Embedding == null || response.Embedding.Count == 0)
    {
        Console.WriteLine("Error: No embedding returned from Ollama.");
        return;
    }

    var queryVector = new Vector(response.Embedding.Select(x => (float)x).ToArray());

    // --- Step 2: Query with cosine distance (<=>) ---
    const int topK = 10;
    var results = new List<(int Id, string SearchText, double Distance)>();

    await using (var cmd = new NpgsqlCommand(
        @"SELECT id, search_text, embedding <=> $1 AS distance
          FROM documents
          WHERE embedding IS NOT NULL
          ORDER BY distance
          LIMIT $2", conn))
    {
        cmd.Parameters.AddWithValue(queryVector);
        cmd.Parameters.AddWithValue(topK);

        await using var reader = await cmd.ExecuteReaderAsync();
        while (await reader.ReadAsync())
        {
            var id = reader.GetInt32(0);
            var searchText = reader.GetString(1);
            var distance = reader.GetDouble(2);
            results.Add((id, searchText, distance));
        }
    }

    // --- Step 3: Display results in table format ---
    if (results.Count == 0)
    {
        Console.WriteLine("No results found.");
        return;
    }

    Console.WriteLine($"\nTop {results.Count} results:\n");

    // Calculate column widths
    const int idWidth = 6;
    const int distWidth = 12;
    const int scoreWidth = 10;
    int textWidth = Math.Min(80, results.Max(r => r.SearchText.Length));
    textWidth = Math.Max(textWidth, 20);

    // Header
    var separator = new string('-', idWidth + textWidth + distWidth + scoreWidth + 13);
    Console.WriteLine(separator);
    Console.WriteLine(string.Format("| {0,-" + idWidth + "} | {1,-" + textWidth + "} | {2,-" + distWidth + "} | {3,-" + scoreWidth + "} |",
        "ID", "Search Text", "Distance", "Score"));
    Console.WriteLine(separator);

    // Rows
    foreach (var (id, searchText, distance) in results)
    {
        var truncated = searchText.Length > textWidth
            ? searchText[..(textWidth - 3)] + "..."
            : searchText;
        var score = 1.0 - distance; // cosine similarity = 1 - cosine distance
        Console.WriteLine(string.Format("| {0,-" + idWidth + "} | {1,-" + textWidth + "} | {2,-" + distWidth + ":F6} | {3,-" + scoreWidth + ":F6} |",
            id, truncated, distance, score));
    }

    Console.WriteLine(separator);
}

// --- Migrate: read documents, embed, and update ---
async Task RunMigrate()
{
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
}
