using MediatR;
using TokenMiner.Application.Mining.Abstractions;
using TokenMiner.Application.Mining.Models;
using TokenMiner.Application.Mining.Services;

namespace TokenMiner.Application.Mining.Sessions;

/// <summary>
/// The caller's in-flight session for a device, if any.
///
/// A client that restarted (or lost its local state) can adopt the session the API still holds
/// instead of being blocked by "already active", and resume reporting liveness for it.
/// </summary>
public sealed record GetMiningSessionQuery(Guid UserId, Guid HardwareId) : IRequest<MiningSessionDto?>;

internal sealed class GetMiningSessionQueryHandler(
    IUserHardwareRepository hardware,
    IMiningSessionRepository sessions,
    IPoolRepository pools,
    ICoinRepository coins,
    IMiningAlgoRepository algorithms,
    MiningOptions miningOptions) : IRequestHandler<GetMiningSessionQuery, MiningSessionDto?>
{
    public async Task<MiningSessionDto?> Handle(
        GetMiningSessionQuery request,
        CancellationToken cancellationToken)
    {
        var device = await hardware.GetByUserAndHardwareIdAsync(
            request.UserId,
            request.HardwareId,
            cancellationToken);

        if (device is null)
        {
            return null;
        }

        var session = await sessions.GetActiveByHardwareAsync(device.Id, cancellationToken);
        if (session is null)
        {
            return null;
        }

        var pool = await pools.GetByIdAsync(session.PoolId, cancellationToken);
        var poolDetails = await pools.GetDetailsAsync(session.PoolId, cancellationToken);
        var coin = await coins.GetByIdAsync(session.CoinId, cancellationToken);
        var algorithm = await algorithms.GetByIdAsync(session.MiningAlgoId, cancellationToken);

        if (pool is null || poolDetails is null || coin is null || algorithm is null)
        {
            // A session whose catalogue rows were removed cannot be described; treat it as absent.
            return null;
        }

        return MiningSessionProjection.Build(
            session,
            device,
            pool,
            poolDetails,
            coin,
            algorithm,
            miningOptions);
    }
}
