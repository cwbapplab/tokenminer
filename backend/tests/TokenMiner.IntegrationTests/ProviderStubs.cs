using TokenMiner.Application.Providers.Abstractions;

namespace TokenMiner.IntegrationTests;

/// <summary>
/// Controllable stand-in for an LLM gateway, so the deposit pipeline can be driven without
/// calling router.one.
/// </summary>
public sealed class StubLlmProviderClient : ILlmProviderClient
{
    public string Name => "router-one";

    public LlmBalance Balance { get; set; } = new(0m, 0m, 0m);

    public List<LlmCatalogModel> Models { get; } = [];

    public int BalanceReads { get; private set; }

    public void Reset()
    {
        Balance = new LlmBalance(0m, 0m, 0m);
        Models.Clear();
        BalanceReads = 0;
    }

    public Task<LlmBalance> GetBalanceAsync(
        LlmProviderContext context,
        CancellationToken cancellationToken)
    {
        BalanceReads++;
        return Task.FromResult(Balance);
    }

    public Task<IReadOnlyList<LlmCatalogModel>> GetModelsAsync(
        LlmProviderContext context,
        CancellationToken cancellationToken) =>
        Task.FromResult<IReadOnlyList<LlmCatalogModel>>(Models.ToList());
}
