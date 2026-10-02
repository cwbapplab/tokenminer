using TokenMiner.Application.Treasury.Abstractions;

namespace TokenMiner.IntegrationTests;

/// <summary>
/// Controllable stand-in for a pool payout integration, so the treasury pipeline can be driven
/// without talking to a real pool.
/// </summary>
public sealed class StubPoolPayoutProvider : IPoolPayoutProvider
{
    /// <summary>Matches the default <c>pools.provider</c>.</summary>
    public string Name => "kryptex";

    public PoolBalance Balance { get; set; } = new(0m, 0m, 0m, null);

    public List<PoolPayoutRecord> Payouts { get; } = [];

    /// <summary>Whether the pool is willing to place a requested withdrawal.</summary>
    public bool PlaceWithdrawals { get; set; }

    public int WithdrawalRequests { get; private set; }

    /// <summary>Which pools a withdrawal was requested for, so tests can assert per pool.</summary>
    public List<Guid> WithdrawalPoolIds { get; } = [];

    public void Reset()
    {
        Balance = new PoolBalance(0m, 0m, 0m, null);
        Payouts.Clear();
        PlaceWithdrawals = false;
        WithdrawalRequests = 0;
        WithdrawalPoolIds.Clear();
    }

    public Task<PoolBalance> GetBalanceAsync(PoolPayoutContext context, CancellationToken cancellationToken) =>
        Task.FromResult(Balance);

    public Task<IReadOnlyList<PoolPayoutRecord>> GetPayoutsAsync(
        PoolPayoutContext context,
        CancellationToken cancellationToken) =>
        Task.FromResult<IReadOnlyList<PoolPayoutRecord>>(Payouts.ToList());

    public Task<PoolWithdrawalResult> RequestWithdrawalAsync(
        PoolPayoutContext context,
        decimal amount,
        CancellationToken cancellationToken)
    {
        WithdrawalRequests++;
        WithdrawalPoolIds.Add(context.PoolId);

        return Task.FromResult(PlaceWithdrawals
            ? new PoolWithdrawalResult(true, null, "stub placed the withdrawal")
            : new PoolWithdrawalResult(false, null, "stub refused the withdrawal"));
    }
}

public sealed class StubCoinPriceProvider : ICoinPriceProvider
{
    public string Source => "stub";

    public decimal? Price { get; set; }

    public void Reset() => Price = null;

    public Task<decimal?> GetUsdPriceAsync(string coinCode, CancellationToken cancellationToken) =>
        Task.FromResult(Price);
}
