using System.Net.Http.Json;
using System.Text.Json;
using System.Text.Json.Serialization;
using Npgsql;
using Qdrant.Client;
using Qdrant.Client.Grpc;

// ========== Configuration from req.md ==========
const string PgHost = "localhost";
const int PgPort = 5432;
const string PgDatabase = "mydb";
const string PgUser = "user";
const string PgPassword = "password";

const string OllamaBaseUrl = "http://localhost:11434";
const string EmbeddingModel = "bge-m3";

const string QdrantGrpcHost = "localhost";
const int QdrantGrpcPort = 6334;
const string QdrantRestUrl = "http://localhost:6333";
const string QdrantApiKey = "demo";
const string CollectionName = "xyz";

// ========== Parse CLI arguments ==========
string? processArg = null;
for (int a = 0; a < args.Length - 1; a++)
{
    if (args[a] == "--process")
    {
        processArg = args[a + 1];
        break;
    }
}

if (string.IsNullOrWhiteSpace(processArg))
{
    Console.WriteLine("Usage: dotnet run -- --process <command>");
    Console.WriteLine();
    Console.WriteLine("Commands:");
    Console.WriteLine("  migrate   Migrate data from PostgreSQL to Qdrant");
    Console.WriteLine("  search    Search in Qdrant collection");
    return;
}

// ========== Shared clients ==========
var connStr = $"Host={PgHost};Port={PgPort};Database={PgDatabase};Username={PgUser};Password={PgPassword}";
using var httpClient = new HttpClient { BaseAddress = new Uri(OllamaBaseUrl), Timeout = TimeSpan.FromMinutes(10) };
var qdrantClient = new QdrantClient(QdrantGrpcHost, QdrantGrpcPort, apiKey: QdrantApiKey);

switch (processArg.ToLower())
{
    case "migrate":
        await RunMigrate();
        break;
    case "migrate-with-rest":
        await RunMigrateWithRest();
        break;
    case "search":
        await RunSearch();
        break;
    default:
        Console.WriteLine($"Unknown command: {processArg}");
        Console.WriteLine("Available commands: migrate, migrate-with-rest, search");
        break;
}

return;

// ========== Migrate with REST API: PostgreSQL -> Qdrant ==========
async Task RunMigrateWithRest()
{
    Console.WriteLine("=== Migration (REST API): PostgreSQL -> Qdrant ===");
    Console.WriteLine();

    // Qdrant REST HttpClient
    using var qdrantHttp = new HttpClient { BaseAddress = new Uri(QdrantRestUrl) };
    qdrantHttp.DefaultRequestHeaders.Add("api-key", QdrantApiKey);

    // Step 1: Read data from PostgreSQL
    var documents = new List<Document>();

    Console.WriteLine("[1/4] Reading data from PostgreSQL...");
    await using (var conn = new NpgsqlConnection(connStr))
    {
        await conn.OpenAsync();
        await using var cmd = new NpgsqlCommand("SELECT id, doc_name, doc_desc, search_text FROM documents", conn);
        await using var reader = await cmd.ExecuteReaderAsync();

        while (await reader.ReadAsync())
        {
            documents.Add(new Document
            {
                Id = reader.GetInt32(0),
                DocName = reader.IsDBNull(1) ? "" : reader.GetString(1),
                DocDes = reader.IsDBNull(2) ? "" : reader.GetString(2),
                SearchText = reader.IsDBNull(3) ? "" : reader.GetString(3)
            });
        }
    }
    Console.WriteLine($"   Found {documents.Count} documents.");

    if (documents.Count == 0)
    {
        Console.WriteLine("   No documents found. Exiting.");
        return;
    }

    // Step 2: Generate embeddings via Ollama
    Console.WriteLine("[2/4] Generating embeddings via Ollama (bge-m3)...");

    var embeddings = new Dictionary<int, float[]>();
    int embeddingDimension = 0;

    for (int i = 0; i < documents.Count; i++)
    {
        var doc = documents[i];
        var text = string.IsNullOrWhiteSpace(doc.SearchText) ? "(empty)" : doc.SearchText;

        var requestBody = new { model = EmbeddingModel, input = text };
        var response = await httpClient.PostAsJsonAsync("/api/embed", requestBody);
        response.EnsureSuccessStatusCode();

        var json = await response.Content.ReadAsStringAsync();
        var result = JsonSerializer.Deserialize<OllamaEmbedResponse>(json);

        if (result?.Embeddings == null || result.Embeddings.Count == 0)
        {
            Console.WriteLine($"   WARNING: No embedding returned for doc id={doc.Id}, skipping.");
            continue;
        }

        var vector = result.Embeddings[0];
        embeddings[doc.Id] = vector;
        embeddingDimension = vector.Length;

        Console.WriteLine($"   [{i + 1}/{documents.Count}] id={doc.Id} embedded (dim={vector.Length})");
    }

    Console.WriteLine($"   Embedding dimension: {embeddingDimension}");

    if (embeddingDimension == 0)
    {
        Console.WriteLine("   No embeddings generated. Exiting.");
        return;
    }

    // Step 3: Create Qdrant collection via REST
    Console.WriteLine("[3/4] Setting up Qdrant collection (REST)...");

    // Check if collection exists
    var checkResp = await qdrantHttp.GetAsync($"/collections/{CollectionName}");
    if (checkResp.IsSuccessStatusCode)
    {
        Console.WriteLine($"   Collection '{CollectionName}' exists. Deleting...");
        var delResp = await qdrantHttp.DeleteAsync($"/collections/{CollectionName}");
        delResp.EnsureSuccessStatusCode();
    }

    // Create collection
    var createBody = new
    {
        vectors = new
        {
            size = embeddingDimension,
            distance = "Cosine"
        }
    };
    var createResp = await qdrantHttp.PutAsJsonAsync($"/collections/{CollectionName}", createBody);
    createResp.EnsureSuccessStatusCode();
    Console.WriteLine($"   Collection '{CollectionName}' created with dimension={embeddingDimension}, distance=Cosine.");

    // Step 4: Upsert data into Qdrant via REST
    Console.WriteLine("[4/4] Upserting data into Qdrant (REST)...");

    var pointsList = new List<object>();
    foreach (var doc in documents)
    {
        if (!embeddings.ContainsKey(doc.Id)) continue;

        pointsList.Add(new
        {
            id = doc.Id,
            vector = embeddings[doc.Id],
            payload = new Dictionary<string, object>
            {
                ["doc_name"] = doc.DocName,
                ["doc_des"] = doc.DocDes,
                ["search_text"] = doc.SearchText
            }
        });
    }

    // Upsert in batches of 100
    const int batchSize = 100;
    for (int i = 0; i < pointsList.Count; i += batchSize)
    {
        var batch = pointsList.Skip(i).Take(batchSize).ToList();
        var upsertBody = new { points = batch };
        var upsertResp = await qdrantHttp.PutAsJsonAsync($"/collections/{CollectionName}/points", upsertBody);
        upsertResp.EnsureSuccessStatusCode();
        Console.WriteLine($"   Upserted batch {i / batchSize + 1} ({batch.Count} points)");
    }

    Console.WriteLine($"   Total points upserted: {pointsList.Count}");
    Console.WriteLine("\nMigration (REST) done!");
}

// ========== Migrate: PostgreSQL -> Qdrant ==========
async Task RunMigrate()
{
    Console.WriteLine("=== Migration: PostgreSQL -> Qdrant ===");
    Console.WriteLine();

    // Step 1: Read data from PostgreSQL
    var documents = new List<Document>();

    Console.WriteLine("[1/4] Reading data from PostgreSQL...");
    await using (var conn = new NpgsqlConnection(connStr))
    {
        await conn.OpenAsync();
        await using var cmd = new NpgsqlCommand("SELECT id, doc_name, doc_desc, search_text FROM documents", conn);
        await using var reader = await cmd.ExecuteReaderAsync();

        while (await reader.ReadAsync())
        {
            documents.Add(new Document
            {
                Id = reader.GetInt32(0),
                DocName = reader.IsDBNull(1) ? "" : reader.GetString(1),
                DocDes = reader.IsDBNull(2) ? "" : reader.GetString(2),
                SearchText = reader.IsDBNull(3) ? "" : reader.GetString(3)
            });
        }
    }
    Console.WriteLine($"   Found {documents.Count} documents.");

    if (documents.Count == 0)
    {
        Console.WriteLine("   No documents found. Exiting.");
        return;
    }

    // Step 2: Generate embeddings via Ollama
    Console.WriteLine("[2/4] Generating embeddings via Ollama (bge-m3)...");

    var embeddings = new Dictionary<int, float[]>();
    int embeddingDimension = 0;

    for (int i = 0; i < documents.Count; i++)
    {
        var doc = documents[i];
        var text = string.IsNullOrWhiteSpace(doc.SearchText) ? "(empty)" : doc.SearchText;

        var requestBody = new { model = EmbeddingModel, input = text };
        var response = await httpClient.PostAsJsonAsync("/api/embed", requestBody);
        response.EnsureSuccessStatusCode();

        var json = await response.Content.ReadAsStringAsync();
        var result = JsonSerializer.Deserialize<OllamaEmbedResponse>(json);

        if (result?.Embeddings == null || result.Embeddings.Count == 0)
        {
            Console.WriteLine($"   WARNING: No embedding returned for doc id={doc.Id}, skipping.");
            continue;
        }

        var vector = result.Embeddings[0];
        embeddings[doc.Id] = vector;
        embeddingDimension = vector.Length;

        Console.WriteLine($"   [{i + 1}/{documents.Count}] id={doc.Id} embedded (dim={vector.Length})");
    }

    Console.WriteLine($"   Embedding dimension: {embeddingDimension}");

    if (embeddingDimension == 0)
    {
        Console.WriteLine("   No embeddings generated. Exiting.");
        return;
    }

    // Step 3: Create Qdrant collection
    Console.WriteLine("[3/4] Setting up Qdrant collection...");

    var collections = await qdrantClient.ListCollectionsAsync();
    if (collections.Any(c => c == CollectionName))
    {
        Console.WriteLine($"   Collection '{CollectionName}' exists. Deleting...");
        await qdrantClient.DeleteCollectionAsync(CollectionName);
    }

    await qdrantClient.CreateCollectionAsync(CollectionName, new VectorParams
    {
        Size = (ulong)embeddingDimension,
        Distance = Distance.Cosine
    });
    Console.WriteLine($"   Collection '{CollectionName}' created with dimension={embeddingDimension}, distance=Cosine.");

    // Step 4: Upsert data into Qdrant
    Console.WriteLine("[4/4] Upserting data into Qdrant...");

    var points = new List<PointStruct>();
    foreach (var doc in documents)
    {
        if (!embeddings.ContainsKey(doc.Id)) continue;

        var point = new PointStruct
        {
            Id = new PointId { Num = (ulong)doc.Id },
            Vectors = embeddings[doc.Id],
        };
        point.Payload.Add("doc_name", doc.DocName);
        point.Payload.Add("doc_des", doc.DocDes);
        point.Payload.Add("search_text", doc.SearchText);

        points.Add(point);
    }

    const int batchSize = 100;
    for (int i = 0; i < points.Count; i += batchSize)
    {
        var batch = points.Skip(i).Take(batchSize).ToList();
        await qdrantClient.UpsertAsync(CollectionName, batch);
        Console.WriteLine($"   Upserted batch {i / batchSize + 1} ({batch.Count} points)");
    }

    Console.WriteLine($"   Total points upserted: {points.Count}");
    Console.WriteLine("\nMigration done!");
}

// ========== Search: Semantic search in Qdrant ==========
async Task RunSearch()
{
    Console.WriteLine("=== Semantic Search ===");
    Console.Write("Enter search query: ");
    var query = Console.ReadLine();

    if (string.IsNullOrWhiteSpace(query))
    {
        Console.WriteLine("No query provided. Exiting.");
        return;
    }

    // Embed the query
    var queryRequest = new { model = EmbeddingModel, input = query };
    var queryResponse = await httpClient.PostAsJsonAsync("/api/embed", queryRequest);
    queryResponse.EnsureSuccessStatusCode();

    var queryJson = await queryResponse.Content.ReadAsStringAsync();
    var queryResult = JsonSerializer.Deserialize<OllamaEmbedResponse>(queryJson);

    if (queryResult?.Embeddings == null || queryResult.Embeddings.Count == 0)
    {
        Console.WriteLine("Failed to generate embedding for query.");
        return;
    }

    var queryVector = queryResult.Embeddings[0];
    var searchResults = await qdrantClient.SearchAsync(CollectionName, queryVector, limit: 5);

    Console.WriteLine($"\nTop {searchResults.Count} results for: \"{query}\"");
    Console.WriteLine(new string('-', 60));

    foreach (var result in searchResults)
    {
        var docName = result.Payload.ContainsKey("doc_name") ? result.Payload["doc_name"].StringValue : "N/A";
        var docDes = result.Payload.ContainsKey("doc_des") ? result.Payload["doc_des"].StringValue : "N/A";
        var searchText = result.Payload.ContainsKey("search_text") ? result.Payload["search_text"].StringValue : "N/A";

        Console.WriteLine($"  Score: {result.Score:F4} | ID: {result.Id.Num}");
        Console.WriteLine($"  Name:  {docName}");
        Console.WriteLine($"  Desc:  {docDes}");
        Console.WriteLine($"  Text:  {(searchText.Length > 100 ? searchText[..100] + "..." : searchText)}");
        Console.WriteLine(new string('-', 60));
    }

    Console.WriteLine("\nDone!");
}

// ========== Models ==========
class Document
{
    public int Id { get; set; }
    public string DocName { get; set; } = "";
    public string DocDes { get; set; } = "";
    public string SearchText { get; set; } = "";
}

class OllamaEmbedResponse
{
    [JsonPropertyName("model")]
    public string? Model { get; set; }

    [JsonPropertyName("embeddings")]
    public List<float[]>? Embeddings { get; set; }
}
