using MediatR;
using TokenMiner.Application.Mining.Models;
using TokenMiner.Contracts.Mining;

namespace TokenMiner.Api.Endpoints;

/// <summary>
/// Signed endpoints consumed by the stratum proxy. Like the share endpoint they authenticate
/// with a service signature rather than a user token.
/// </summary>
public static class InternalStratumEndpoints
{
    public static IEndpointRouteBuilder MapInternalStratumEndpoints(this IEndpointRouteBuilder app)
    {
        ArgumentNullException.ThrowIfNull(app);

        var group = app.MapGroup("/internal/stratum")
            .WithTags("Internal")
            .AllowAnonymous();

        group.MapGet("/config", GetConfigAsync)
            .WithName("GetStratumConfig")
            .Produces<StratumConfigResponse>()
            .ProducesProblem(StatusCodes.Status401Unauthorized);

        group.MapGet("/workers/{workerId}", ResolveWorkerAsync)
            .WithName("ResolveStratumWorker")
            .Produces<StratumWorkerResponse>()
            .ProducesProblem(StatusCodes.Status401Unauthorized)
            .ProducesProblem(StatusCodes.Status404NotFound);

        return app;
    }

    private static async Task<IResult> GetConfigAsync(ISender sender, CancellationToken cancellationToken)
    {
        var config = await sender.Send(new GetStratumConfigQuery(), cancellationToken);

        return Results.Ok(new StratumConfigResponse(
            config.Version,
            config.Pools
                .Select(pool => new StratumPoolConfig(
                    pool.PoolId,
                    pool.SystemPoolId,
                    pool.Name,
                    pool.StratumEndpoint,
                    pool.PayoutAddress,
                    pool.Coins.Select(coin => new StratumCoinConfig(coin.CoinId, coin.Code)).ToList()))
                .ToList()));
    }

    private static async Task<IResult> ResolveWorkerAsync(
        string workerId,
        ISender sender,
        CancellationToken cancellationToken)
    {
        var worker = await sender.Send(new ResolveStratumWorkerQuery(workerId), cancellationToken);

        if (worker is null)
        {
            return Results.Problem(
                detail: $"No active mining session for worker '{workerId}'.",
                statusCode: StatusCodes.Status404NotFound);
        }

        return Results.Ok(new StratumWorkerResponse(
            worker.WorkerId,
            worker.UserId,
            worker.UserHardwareId,
            worker.UserHardwareMinerId,
            worker.PoolId,
            worker.StratumEndpoint,
            worker.PayoutAddress,
            worker.CoinId,
            worker.CoinCode));
    }
}
