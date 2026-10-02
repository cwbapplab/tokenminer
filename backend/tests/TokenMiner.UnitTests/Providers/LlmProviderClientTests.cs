using System.Net;
using System.Text;
using FluentAssertions;
using Microsoft.Extensions.Logging.Abstractions;
using TokenMiner.Application.Providers.Abstractions;
using TokenMiner.Infrastructure.Integrations;
using Xunit;

namespace TokenMiner.UnitTests.Providers;

/// <summary>
/// Pins the parsing against the shapes the gateway actually returns, so a change on their side
/// shows up here rather than as a silent zero balance.
/// </summary>
public sealed class OpenAiCompatibleLlmProviderClientTests
{
    private static readonly LlmProviderContext Context =
        new("router-one", "https://api.router.one/v1", "sk-test");

    private sealed class StubHandler(HttpStatusCode status, string body) : HttpMessageHandler
    {
        public Uri? LastRequest { get; private set; }

        public string? Authorization { get; private set; }

        protected override Task<HttpResponseMessage> SendAsync(
            HttpRequestMessage request,
            CancellationToken cancellationToken)
        {
            LastRequest = request.RequestUri;
            Authorization = request.Headers.Authorization?.ToString();

            return Task.FromResult(new HttpResponseMessage(status)
            {
                Content = new StringContent(body, Encoding.UTF8, "application/json"),
            });
        }
    }

    private static (OpenAiCompatibleLlmProviderClient Client, StubHandler Handler) CreateClient(string body)
    {
        var handler = new StubHandler(HttpStatusCode.OK, body);

        return (
            new OpenAiCompatibleLlmProviderClient(
                new HttpClient(handler),
                NullLogger<OpenAiCompatibleLlmProviderClient>.Instance),
            handler);
    }

    [Fact]
    public async Task ReadsTheBalanceFieldsTheGatewayDocuments()
    {
        var (client, handler) = CreateClient(
            """{"object":"balance","currency":"USD","balance":12.345678,"reserved_balance":0.5,"total_balance":12.845678}""");

        var balance = await client.GetBalanceAsync(Context, CancellationToken.None);

        balance.Balance.Should().Be(12.345678m);
        balance.ReservedBalance.Should().Be(0.5m);
        balance.TotalBalance.Should().Be(12.845678m);

        handler.LastRequest!.AbsolutePath.Should().Be("/v1/balance");
        handler.Authorization.Should().Be("Bearer sk-test");
    }

    [Fact]
    public async Task ReadsTheModelCatalogue()
    {
        const string body = """
            {
              "object": "list",
              "data": [
                {
                  "id": "anthropic/claude-sonnet-5",
                  "object": "model",
                  "capabilities": ["chat", "streaming", "tool_calling", "vision"],
                  "max_tokens": 1048576,
                  "category": "text",
                  "pricing_mode": "token"
                }
              ]
            }
            """;

        var (client, handler) = CreateClient(body);

        var models = await client.GetModelsAsync(Context, CancellationToken.None);

        models.Should().HaveCount(1);
        models[0].ModelId.Should().Be("anthropic/claude-sonnet-5");
        models[0].ContextLength.Should().Be(1048576);
        models[0].Currency.Should().Be("USD");
        models[0].Capabilities.Should().BeEquivalentTo(["chat", "streaming", "tool_calling", "vision"]);

        // The documented catalogue carries no prices, so costs stay unknown rather than zero.
        models[0].InputCost.Should().BeNull();
        models[0].OutputCost.Should().BeNull();

        handler.LastRequest!.AbsolutePath.Should().Be("/v1/models");
    }

    [Fact]
    public async Task PicksUpPricesWhenTheCatalogueProvidesThem()
    {
        const string body = """
            {
              "data": [
                {
                  "id": "vendor/model-x",
                  "name": "Model X",
                  "context_length": 128000,
                  "input_cost": 3.0,
                  "output_cost": 15.0,
                  "pricing": { "cached_input": 0.3 }
                }
              ]
            }
            """;

        var (client, _) = CreateClient(body);

        var models = await client.GetModelsAsync(Context, CancellationToken.None);

        models[0].Name.Should().Be("Model X");
        models[0].InputCost.Should().Be(3.0m);
        models[0].OutputCost.Should().Be(15.0m);
        models[0].CachedInputCost.Should().Be(0.3m);
    }

    [Fact]
    public async Task ReturnsNothingWhenTheCatalogueHasNoDataArray()
    {
        var (client, _) = CreateClient("""{"object":"list"}""");

        var models = await client.GetModelsAsync(Context, CancellationToken.None);

        models.Should().BeEmpty();
    }
}
