using MediatR;
using Microsoft.Extensions.Logging;
using TokenMiner.Application.Common.Abstractions;
using TokenMiner.Application.Mining.Abstractions;
using TokenMiner.Application.Treasury.Abstractions;

namespace TokenMiner.Application.Treasury.Commands;

public sealed record ReconcilePayoutsCommand : IRequest<PayoutReconciliationResult>;

/// <summary>
/// Settles payouts that have enough confirmations and then attributes confirmed funds to the
/// shares that earned them.
/// </summary>
/// <remarks>
/// Attribution is deliberately conservative: a share is only redeemed when its mined amount is
/// known, and the running total may never exceed the payout. A share is therefore redeemed at
/// most once, and re-running the reconciliation changes nothing.
/// </remarks>
internal sealed class ReconcilePayoutsCommandHandler(
    IPoolPayoutRepository payouts,
    IShareRepository shares,
    IUnitOfWork unitOfWork,
    TimeProvider timeProvider,
    TreasuryOptions options,
    ILogger<ReconcilePayoutsCommandHandler> logger)
    : IRequestHandler<ReconcilePayoutsCommand, PayoutReconciliationResult>
{
    public async Task<PayoutReconciliationResult> Handle(
        ReconcilePayoutsCommand request,
        CancellationToken cancellationToken)
    {
        var now = timeProvider.GetUtcNow();

        var confirmed = 0;

        foreach (var payout in await payouts.ListUnsettledAsync(cancellationToken))
        {
            if (payout.ConfirmationsCount < options.PayoutConfirmationThreshold)
            {
                continue;
            }

            if (payout.Confirm(now, now))
            {
                confirmed++;
            }
        }

        var redeemed = 0;

        foreach (var payout in await payouts.ListRedeemableAsync(cancellationToken))
        {
            var remaining = payout.RemainingAmount;
            if (remaining <= 0)
            {
                continue;
            }

            var accepted = await shares.ListAcceptedByPoolAsync(payout.PoolId, cancellationToken);
            var attributed = 0m;

            foreach (var share in accepted)
            {
                var value = share.CoinValue ?? 0m;

                // Without a known amount there is nothing to attribute.
                if (value <= 0)
                {
                    continue;
                }

                // Never attribute more than the payout carried.
                if (attributed + value > remaining)
                {
                    break;
                }

                share.Redeem(now);
                attributed += value;
                redeemed++;
            }

            if (attributed > 0)
            {
                payout.RecordRedeemed(attributed, now);

                logger.LogInformation(
                    "Attributed {Amount} from payout {Payout} to shares.",
                    attributed,
                    payout.Id);
            }
        }

        if (confirmed > 0 || redeemed > 0)
        {
            await unitOfWork.SaveChangesAsync(cancellationToken);
        }

        return new PayoutReconciliationResult(confirmed, redeemed);
    }
}
