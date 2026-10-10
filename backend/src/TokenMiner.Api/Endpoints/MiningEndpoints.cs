using System.Security.Claims;
using MediatR;
using TokenMiner.Application.Mining.Models;
using TokenMiner.Application.Mining.Sessions;
using TokenMiner.Contracts.Mining;

namespace TokenMiner.Api.Endpoints;

public static class MiningEndpoints
{
    public static IEndpointRouteBuilder MapMiningEndpoints(this IEndpointRouteBuilder app)
    {
        ArgumentNullException.ThrowIfNull(app);

        var group = app.MapGroup("/api/mining")
            .WithTags("Mining")
            .RequireAuthorization(AuthorizationPolicies.User);

        group.MapPost("/start", StartAsync)
            .WithName("StartMining")
            .Produces<MiningSessionResponse>()
            .ProducesProblem(StatusCodes.Status400BadRequest)
            .ProducesProblem(StatusCodes.Status409Conflict);

        group.MapPost("/stop", StopAsync)
            .WithName("StopMining")
            .Produces(StatusCodes.Status204NoContent)
            .ProducesProblem(StatusCodes.Status400BadRequest);

        // Lets a client that restarted adopt the session the API still holds for its device.
        group.MapGet("/session", GetSessionAsync)
            .WithName("GetMiningSession")
            .Produces<MiningSessionResponse>()
            .Produces(StatusCodes.Status204NoContent)
            .ProducesProblem(StatusCodes.Status401Unauthorized);

        return app;
    }

    private static async Task<IResult> StartAsync(
        StartMiningRequest request,
        ClaimsPrincipal user,
        ISender sender,
        CancellationToken cancellationToken)
    {
        var userId = user.GetUserId();
        if (userId is null)
        {
            return Results.Unauthorized();
        }

        var session = await sender.Send(
            new StartMiningCommand(userId.Value, request.HardwareId, request.PoolId),
            cancellationToken);

        return Results.Ok(ToResponse(session));
    }

    private static async Task<IResult> GetSessionAsync(
        Guid? hardwareId,
        ClaimsPrincipal user,
        ISender sender,
        CancellationToken cancellationToken)
    {
        var userId = user.GetUserId();
        if (userId is null)
        {
            return Results.Unauthorized();
        }

        if (hardwareId is null || hardwareId == Guid.Empty)
        {
            return Results.BadRequest();
        }

        var session = await sender.Send(
            new GetMiningSessionQuery(userId.Value, hardwareId.Value),
            cancellationToken);

        return session is null
            ? Results.NoContent()
            : Results.Ok(ToResponse(session));
    }

    private static MiningSessionResponse ToResponse(MiningSessionDto session) => new(
        session.SessionId,
        session.PoolId,
        session.PoolName,
        session.CoinId,
        session.CoinCode,
        session.AlgorithmId,
        session.AlgorithmCode,
        session.WorkerId,
        session.MinerCommand,
        session.MinerConfig,
        session.StratumEndpoint,
        session.Status,
        session.StartedAt);

    private static async Task<IResult> StopAsync(
        StopMiningRequest request,
        ClaimsPrincipal user,
        ISender sender,
        CancellationToken cancellationToken)
    {
        var userId = user.GetUserId();
        if (userId is null)
        {
            return Results.Unauthorized();
        }

        await sender.Send(
            new StopMiningCommand(userId.Value, request.HardwareId, request.Reason),
            cancellationToken);

        return Results.NoContent();
    }
}
