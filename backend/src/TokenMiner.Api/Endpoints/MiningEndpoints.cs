using System.Net.WebSockets;
using System.Security.Claims;
using System.Text.Json;
using MediatR;
using TokenMiner.Application.Mining.Models;
using TokenMiner.Application.Mining.Sessions;
using TokenMiner.Contracts.Mining;

namespace TokenMiner.Api.Endpoints;

public static class MiningEndpoints
{
    private const int ReceiveBufferSize = 4096;

    private static readonly JsonSerializerOptions JsonOptions = new(JsonSerializerDefaults.Web);

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

        // Liveness socket. A browser-based client (including a Tauri webview) cannot set an
        // Authorization header on a WebSocket, so the token may also arrive as ?access_token=.
        app.MapGet("/ws/mining", HandleSocketAsync)
            .WithTags("Mining")
            .WithName("MiningActivitySocket")
            .RequireAuthorization(AuthorizationPolicies.User);

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

    private static async Task HandleSocketAsync(HttpContext context, ISender sender)
    {
        if (!context.WebSockets.IsWebSocketRequest)
        {
            context.Response.StatusCode = StatusCodes.Status400BadRequest;
            await context.Response.WriteAsync("A WebSocket upgrade request is required.");
            return;
        }

        var userId = context.User.GetUserId();
        if (userId is null)
        {
            context.Response.StatusCode = StatusCodes.Status401Unauthorized;
            return;
        }

        using var socket = await context.WebSockets.AcceptWebSocketAsync();

        var cancellationToken = context.RequestAborted;
        var buffer = new byte[ReceiveBufferSize];

        try
        {
            while (socket.State == WebSocketState.Open && !cancellationToken.IsCancellationRequested)
            {
                var received = await socket.ReceiveAsync(buffer, cancellationToken);

                if (received.MessageType == WebSocketMessageType.Close)
                {
                    break;
                }

                if (received.MessageType != WebSocketMessageType.Text)
                {
                    continue;
                }

                var heartbeat = TryReadHeartbeat(buffer, received.Count);
                if (heartbeat is null)
                {
                    await SendAsync(
                        socket,
                        new { type = "error", message = "Expected {\"type\":\"heartbeat\",\"hardwareId\":\"<guid>\"}." },
                        cancellationToken);

                    continue;
                }

                var result = await sender.Send(
                    new RecordMiningHeartbeatCommand(userId.Value, heartbeat.HardwareId),
                    cancellationToken);

                await SendAsync(
                    socket,
                    new MiningHeartbeatAck("ack", result.SessionFound, result.Status, result.Resumed, result.At),
                    cancellationToken);
            }
        }
        catch (OperationCanceledException)
        {
            // The client went away; the watchdog pauses the session once the grace period lapses.
        }
        catch (WebSocketException)
        {
            // Abrupt disconnect, same handling as above.
        }

        await CloseQuietlyAsync(socket);
    }

    private static MiningHeartbeatMessage? TryReadHeartbeat(byte[] buffer, int count)
    {
        try
        {
            var message = JsonSerializer.Deserialize<MiningHeartbeatMessage>(buffer.AsSpan(0, count), JsonOptions);

            return message is not null && message.HardwareId != Guid.Empty ? message : null;
        }
        catch (JsonException)
        {
            return null;
        }
    }

    private static Task SendAsync(WebSocket socket, object payload, CancellationToken cancellationToken)
    {
        var bytes = JsonSerializer.SerializeToUtf8Bytes(payload, JsonOptions);

        return socket.SendAsync(bytes, WebSocketMessageType.Text, endOfMessage: true, cancellationToken);
    }

    private static async Task CloseQuietlyAsync(WebSocket socket)
    {
        if (socket.State != WebSocketState.Open)
        {
            return;
        }

        try
        {
            await socket.CloseAsync(WebSocketCloseStatus.NormalClosure, statusDescription: null, CancellationToken.None);
        }
        catch (WebSocketException)
        {
            // Already gone.
        }
    }
}
