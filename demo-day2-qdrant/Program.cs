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
const string QdrantApiKey = "demo";
const string CollectionName = "xyz";

// ========== Step 1: Read data from PostgreSQL ==========
Console.WriteLine("=== Migration: PostgreSQL -> Qdrant ===");
Console.WriteLine();

var connStr = $"Host={PgHost};Port={PgPort};Database={PgDatabase};Username={PgUser};Password={PgPassword}";
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

// ========== Step 2: Generate embeddings via Ollama ==========
Console.WriteLine("[2/4] Generating embeddings via Ollama (bge-m3)...");

using var httpClient = new HttpClient { BaseAddress = new Uri(OllamaBaseUrl), Timeout = TimeSpan.FromMinutes(10) };
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

// ========== Step 3: Create Qdrant collection ==========
Console.WriteLine("[3/4] Setting up Qdrant collection...");

var qdrantClient = new QdrantClient(QdrantGrpcHost, QdrantGrpcPort, apiKey: QdrantApiKey);

// Check if collection exists, recreate it
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

// ========== Step 4: Upsert data into Qdrant ==========
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

// Upsert in batches of 100
const int batchSize = 100;
for (int i = 0; i < points.Count; i += batchSize)
{
    var batch = points.Skip(i).Take(batchSize).ToList();
    await qdrantClient.UpsertAsync(CollectionName, batch);
    Console.WriteLine($"   Upserted batch {i / batchSize + 1} ({batch.Count} points)");
}

Console.WriteLine($"   Total points upserted: {points.Count}");
Console.WriteLine();

// ========== Step 5: Demo search ==========
Console.WriteLine("=== Demo: Semantic Search ===");
Console.Write("Enter search query (or press Enter to skip): ");
var query = Console.ReadLine();

if (!string.IsNullOrWhiteSpace(query))
{
    // Embed the query
    var queryRequest = new { model = EmbeddingModel, input = query };
    var queryResponse = await httpClient.PostAsJsonAsync("/api/embed", queryRequest);
    queryResponse.EnsureSuccessStatusCode();

    var queryJson = await queryResponse.Content.ReadAsStringAsync();
    var queryResult = JsonSerializer.Deserialize<OllamaEmbedResponse>(queryJson);

    if (queryResult?.Embeddings != null && queryResult.Embeddings.Count > 0)
    {
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
    }
}

Console.WriteLine("\nDone!");

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
