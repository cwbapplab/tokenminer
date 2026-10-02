using MediatR;
using TokenMiner.Application.Mining.Shares;
using TokenMiner.Contracts.Mining;

namespace TokenMiner.Api.Endpoints;

/// <summary>
/// Machine-to-machine endpoints called by the stratum proxy. These are deliberately not
/// reachable with a user token: they authenticate with a signed service request instead.
/// </summary>
public static class InternalMiningEndpoints
{
    public static IEndpointRouteBuilder MapInternalMiningEndpoints(this IEndpointRouteBuilder app)
    {
        ArgumentNullException.ThrowIfNull(app);

        // Service-authenticated, not user-authenticated: the signature is verified by
        // ServiceAuthMiddleware, which runs before endpoint binding.
        var group = app.MapGroup("/internal/mining")
            .WithTags("Internal")
            .AllowAnonymous();

        group.MapPost("/shares", RecordShareAsync)
            .WithName("RecordMiningShare")
            .Produces<MiningShareResponse>()
            .ProducesProblem(StatusCodes.Status400BadRequest)
            .ProducesProblem(StatusCodes.Status401Unauthorized);

        return app;
    }

    private static async Task<IResult> RecordShareAsync(
        MiningShareRequest request,
        ISender sender,
        CancellationToken cancellationToken)
    {
        var recorded = await sender.Send(
            new RecordShareCommand(
                request.UserId,
                request.UserHardwareId,
                request.PoolId,
                request.CoinId,
                request.ShareIdentifier,
                request.JobId,
                request.Nonce,
                request.Extranonce,
                request.Difficulty,
                request.Target,
                request.Hash,
                request.Timestamp,
                request.CoinValue,
                request.PoolResponse?.GetRawText()),
            cancellationToken);

        return Results.Ok(new MiningShareResponse(
            recorded.Id,
            recorded.ShareIdentifier,
            recorded.RewardStatus,
            recorded.CoinValue,
            recorded.ApproxUsdValueAtTime,
            recorded.AlreadyRecorded,
            recorded.CreatedAt));
    }
}
