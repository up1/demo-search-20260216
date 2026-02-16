using System.Text;
using System.Text.Json;

var ollamaUrl = "http://152.42.202.40:11434/api/embed";
var model = "bge-m3";

Console.Write("Enter text to embed: ");
var input = Console.ReadLine();

if (string.IsNullOrWhiteSpace(input))
{
    Console.WriteLine("No input provided.");
    return;
}

using var httpClient = new HttpClient();
httpClient.Timeout = TimeSpan.FromSeconds(60);

var requestBody = new
{
    model = model,
    input = input
};

var json = JsonSerializer.Serialize(requestBody);
var content = new StringContent(json, Encoding.UTF8, "application/json");

Console.WriteLine($"\nSending text to Ollama ({model})...\n");

var response = await httpClient.PostAsync(ollamaUrl, content);
var responseBody = await response.Content.ReadAsStringAsync();

if (!response.IsSuccessStatusCode)
{
    Console.WriteLine($"Error: {response.StatusCode}");
    Console.WriteLine(responseBody);
    return;
}

using var doc = JsonDocument.Parse(responseBody);
var embeddings = doc.RootElement.GetProperty("embeddings");

// Get the first embedding vector
var vector = embeddings[0];
var values = new List<double>();
foreach (var val in vector.EnumerateArray())
{
    values.Add(val.GetDouble());
}

Console.WriteLine($"Vector dimension: {values.Count}");
Console.WriteLine($"First 10 values: [{string.Join(", ", values.Take(10).Select(v => v.ToString("F6")))}]");
Console.WriteLine($"\nFull vector:\n[{string.Join(", ", values.Select(v => v.ToString("F6")))}]");
