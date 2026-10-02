using MediatR;
using Microsoft.Extensions.Logging;
using TokenMiner.Application.Common.Abstractions;
using TokenMiner.Application.Mining.Abstractions;
using TokenMiner.Application.Treasury.Abstractions;
using TokenMiner.Domain.Mining;
using TokenMiner.Domain.Mining.Enums;

namespace TokenMiner.Application.Treasury.Commands;

public sealed record RefreshCoinPricesCommand : IRequest<int>;

/// <summary>
/// Refreshes each active coin's cached USD price and appends a history point. History is never
/// overwritten, so past share valuations stay reproducible.
/// </summary>
internal sealed class RefreshCoinPricesCommandHandler(
    ICoinRepository coins,
    ICoinPriceProvider prices,
    IUnitOfWork unitOfWork,
    TimeProvider timeProvider,
    ILogger<RefreshCoinPricesCommandHandler> logger)
    : IRequestHandler<RefreshCoinPricesCommand, int>
{
    public async Task<int> Handle(RefreshCoinPricesCommand request, CancellationToken cancellationToken)
    {
        var now = timeProvider.GetUtcNow();
        var updated = 0;

        foreach (var coin in (await coins.ListAsync(cancellationToken))
                     .Where(coin => coin.Status == MiningStatus.Active))
        {
            decimal? price;

            try
            {
                price = await prices.GetUsdPriceAsync(coin.Code, cancellationToken);
            }
            catch (Exception exception)
            {
                // A missing price is not fatal: the previous value simply stays in place.
                logger.LogWarning(exception, "Fetching the price for {Coin} failed.", coin.Code);
                continue;
            }

            if (price is null || price <= 0)
            {
                continue;
            }

            coin.RecordPrice(price.Value, now, now);

            coins.AddPriceHistory(new CoinPriceHistory(
                Guid.NewGuid(),
                coin.Id,
                price.Value,
                now,
                prices.Source));

            updated++;
        }

        if (updated > 0)
        {
            await unitOfWork.SaveChangesAsync(cancellationToken);
        }

        return updated;
    }
}
