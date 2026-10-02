using MediatR;
using TokenMiner.Application.Common.Abstractions;
using TokenMiner.Application.Mining.Abstractions;

namespace TokenMiner.Application.Mining.Shares;

public sealed record ProcessPendingSharesCommand(int BatchSize) : IRequest<int>;

/// <summary>
/// Advances recorded shares from <c>pending</c> to <c>accepted</c>. A share whose pool/coin
/// relationship no longer holds is rejected outright, because that is a data-integrity fault
/// rather than a transient one. Shares are moved to <c>redeemed</c> later, when a confirmed
/// pool payout covers them.
/// </summary>
internal sealed class ProcessPendingSharesCommandHandler(
    IShareRepository shares,
    IPoolRepository pools,
    IUnitOfWork unitOfWork,
    TimeProvider timeProvider) : IRequestHandler<ProcessPendingSharesCommand, int>
{
    public async Task<int> Handle(ProcessPendingSharesCommand request, CancellationToken cancellationToken)
    {
        var batchSize = Math.Clamp(request.BatchSize, 1, 1000);

        var pending = await shares.ListPendingAsync(batchSize, cancellationToken);
        if (pending.Count == 0)
        {
            return 0;
        }

        var now = timeProvider.GetUtcNow();
        var processed = 0;

        foreach (var share in pending)
        {
            if (!share.MarkProcessing())
            {
                continue;
            }

            if (!await pools.SupportsCoinAsync(share.PoolId, share.CoinId, cancellationToken))
            {
                share.Reject("pool_coin_relationship_invalid", now);
                processed++;
                continue;
            }

            share.Accept(now);
            processed++;
        }

        if (processed > 0)
        {
            await unitOfWork.SaveChangesAsync(cancellationToken);
        }

        return processed;
    }
}
