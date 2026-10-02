using System.Text.Json;
using MediatR;
using TokenMiner.Application.Mining.Algorithms;
using TokenMiner.Application.Mining.Coins;
using TokenMiner.Application.Mining.Models;
using TokenMiner.Application.Mining.Pools;
using TokenMiner.Contracts.Mining;

namespace TokenMiner.Api.Endpoints;

/// <summary>
/// Administrative catalogue management for coins, pools and mining algorithms. All routes
/// require the <c>admin</c> role.
/// </summary>
public static class AdminMiningEndpoints
{
    public static IEndpointRouteBuilder MapAdminMiningEndpoints(this IEndpointRouteBuilder app)
    {
        ArgumentNullException.ThrowIfNull(app);

        MapCoins(app);
        MapPools(app);
        MapMiningAlgos(app);

        return app;
    }

    private static void MapCoins(IEndpointRouteBuilder app)
    {
        var coins = app.MapGroup("/api/admin/coins")
            .WithTags("Admin - Coins")
            .RequireAuthorization(AuthorizationPolicies.Admin);

        coins.MapGet(string.Empty, ListCoinsAsync)
            .WithName("AdminListCoins")
            .Produces<IReadOnlyList<CoinResponse>>();

        coins.MapGet("/{id:guid}", GetCoinAsync)
            .WithName("AdminGetCoin")
            .Produces<CoinResponse>()
            .ProducesProblem(StatusCodes.Status404NotFound);

        coins.MapPost(string.Empty, CreateCoinAsync)
            .WithName("AdminCreateCoin")
            .Produces<CoinResponse>(StatusCodes.Status201Created)
            .ProducesProblem(StatusCodes.Status400BadRequest)
            .ProducesProblem(StatusCodes.Status409Conflict);

        coins.MapPut("/{id:guid}", UpdateCoinAsync)
            .WithName("AdminUpdateCoin")
            .Produces<CoinResponse>()
            .ProducesProblem(StatusCodes.Status400BadRequest)
            .ProducesProblem(StatusCodes.Status404NotFound);
    }

    private static void MapPools(IEndpointRouteBuilder app)
    {
        var pools = app.MapGroup("/api/admin/pools")
            .WithTags("Admin - Pools")
            .RequireAuthorization(AuthorizationPolicies.Admin);

        pools.MapGet(string.Empty, ListPoolsAsync)
            .WithName("AdminListPools")
            .Produces<IReadOnlyList<PoolResponse>>();

        pools.MapGet("/{id:guid}", GetPoolAsync)
            .WithName("AdminGetPool")
            .Produces<PoolResponse>()
            .ProducesProblem(StatusCodes.Status404NotFound);

        pools.MapPost(string.Empty, CreatePoolAsync)
            .WithName("AdminCreatePool")
            .Produces<PoolResponse>(StatusCodes.Status201Created)
            .ProducesProblem(StatusCodes.Status400BadRequest)
            .ProducesProblem(StatusCodes.Status409Conflict);

        pools.MapPut("/{id:guid}", UpdatePoolAsync)
            .WithName("AdminUpdatePool")
            .Produces<PoolResponse>()
            .ProducesProblem(StatusCodes.Status400BadRequest)
            .ProducesProblem(StatusCodes.Status404NotFound);
    }

    private static void MapMiningAlgos(IEndpointRouteBuilder app)
    {
        var algorithms = app.MapGroup("/api/admin/mining-algos")
            .WithTags("Admin - Mining algorithms")
            .RequireAuthorization(AuthorizationPolicies.Admin);

        algorithms.MapGet(string.Empty, ListMiningAlgosAsync)
            .WithName("AdminListMiningAlgos")
            .Produces<IReadOnlyList<MiningAlgoResponse>>();

        algorithms.MapGet("/{id:guid}", GetMiningAlgoAsync)
            .WithName("AdminGetMiningAlgo")
            .Produces<MiningAlgoResponse>()
            .ProducesProblem(StatusCodes.Status404NotFound);

        algorithms.MapPost(string.Empty, CreateMiningAlgoAsync)
            .WithName("AdminCreateMiningAlgo")
            .Produces<MiningAlgoResponse>(StatusCodes.Status201Created)
            .ProducesProblem(StatusCodes.Status400BadRequest)
            .ProducesProblem(StatusCodes.Status409Conflict);

        algorithms.MapPut("/{id:guid}", UpdateMiningAlgoAsync)
            .WithName("AdminUpdateMiningAlgo")
            .Produces<MiningAlgoResponse>()
            .ProducesProblem(StatusCodes.Status400BadRequest)
            .ProducesProblem(StatusCodes.Status404NotFound);
    }

    // --- Coins ---------------------------------------------------------------------------

    private static async Task<IResult> ListCoinsAsync(ISender sender, CancellationToken cancellationToken)
    {
        var coins = await sender.Send(new ListCoinsQuery(), cancellationToken);
        return Results.Ok(coins.Select(ToResponse).ToList());
    }

    private static async Task<IResult> GetCoinAsync(Guid id, ISender sender, CancellationToken cancellationToken)
    {
        var coin = await sender.Send(new GetCoinQuery(id), cancellationToken);
        return Results.Ok(ToResponse(coin));
    }

    private static async Task<IResult> CreateCoinAsync(
        CreateCoinRequest request,
        ISender sender,
        CancellationToken cancellationToken)
    {
        var coin = await sender.Send(
            new CreateCoinCommand(request.Code, request.Name, request.Network, request.Decimals),
            cancellationToken);

        return Results.Created($"/api/admin/coins/{coin.Id}", ToResponse(coin));
    }

    private static async Task<IResult> UpdateCoinAsync(
        Guid id,
        UpdateCoinRequest request,
        ISender sender,
        CancellationToken cancellationToken)
    {
        var coin = await sender.Send(
            new UpdateCoinCommand(id, request.Name, request.Network, request.Decimals, request.Status),
            cancellationToken);

        return Results.Ok(ToResponse(coin));
    }

    // --- Pools ---------------------------------------------------------------------------

    private static async Task<IResult> ListPoolsAsync(ISender sender, CancellationToken cancellationToken)
    {
        var pools = await sender.Send(new ListPoolsQuery(), cancellationToken);
        return Results.Ok(pools.Select(ToResponse).ToList());
    }

    private static async Task<IResult> GetPoolAsync(Guid id, ISender sender, CancellationToken cancellationToken)
    {
        var pool = await sender.Send(new GetPoolQuery(id), cancellationToken);
        return Results.Ok(ToResponse(pool));
    }

    private static async Task<IResult> CreatePoolAsync(
        CreatePoolRequest request,
        ISender sender,
        CancellationToken cancellationToken)
    {
        var pool = await sender.Send(
            new CreatePoolCommand(request.SystemPoolId, request.Name, request.Provider, ToInput(request.Details), request.CoinIds),
            cancellationToken);

        return Results.Created($"/api/admin/pools/{pool.Id}", ToResponse(pool));
    }

    private static async Task<IResult> UpdatePoolAsync(
        Guid id,
        UpdatePoolRequest request,
        ISender sender,
        CancellationToken cancellationToken)
    {
        var pool = await sender.Send(
            new UpdatePoolCommand(id, request.Name, request.Provider, request.Status, ToInput(request.Details), request.CoinIds),
            cancellationToken);

        return Results.Ok(ToResponse(pool));
    }

    // --- Mining algorithms ---------------------------------------------------------------

    private static async Task<IResult> ListMiningAlgosAsync(ISender sender, CancellationToken cancellationToken)
    {
        var algorithms = await sender.Send(new ListMiningAlgosQuery(), cancellationToken);
        return Results.Ok(algorithms.Select(ToResponse).ToList());
    }

    private static async Task<IResult> GetMiningAlgoAsync(Guid id, ISender sender, CancellationToken cancellationToken)
    {
        var algorithm = await sender.Send(new GetMiningAlgoQuery(id), cancellationToken);
        return Results.Ok(ToResponse(algorithm));
    }

    private static async Task<IResult> CreateMiningAlgoAsync(
        CreateMiningAlgoRequest request,
        ISender sender,
        CancellationToken cancellationToken)
    {
        var algorithm = await sender.Send(
            new CreateMiningAlgoCommand(request.Code, request.Name, request.Priority, ToRawJson(request.Configuration)),
            cancellationToken);

        return Results.Created($"/api/admin/mining-algos/{algorithm.Id}", ToResponse(algorithm));
    }

    private static async Task<IResult> UpdateMiningAlgoAsync(
        Guid id,
        UpdateMiningAlgoRequest request,
        ISender sender,
        CancellationToken cancellationToken)
    {
        var algorithm = await sender.Send(
            new UpdateMiningAlgoCommand(id, request.Name, request.Priority, request.Status, ToRawJson(request.Configuration)),
            cancellationToken);

        return Results.Ok(ToResponse(algorithm));
    }

    // --- Mapping -------------------------------------------------------------------------

    private static CoinResponse ToResponse(CoinDto coin) => new(
        coin.Id,
        coin.Code,
        coin.Name,
        coin.Network,
        coin.Decimals,
        coin.LastKnownUsdValue,
        coin.LastValueDate,
        coin.Status,
        coin.CreatedAt,
        coin.UpdatedAt);

    private static PoolResponse ToResponse(PoolDto pool) => new(
        pool.Id,
        pool.SystemPoolId,
        pool.Name,
        pool.Provider,
        pool.Status,
        pool.Details is null
            ? null
            : new PoolDetailsResponse(
                pool.Details.BaseUrl,
                pool.Details.StatusEndpoint,
                pool.Details.StratumEndpoint,
                pool.Details.CoinId,
                pool.Details.PayoutMode,
                pool.Details.PayoutAddress,
                pool.Details.PayoutNetwork),
        pool.CoinIds,
        pool.CreatedAt,
        pool.UpdatedAt);

    private static MiningAlgoResponse ToResponse(MiningAlgoDto algorithm) => new(
        algorithm.Id,
        algorithm.Code,
        algorithm.Name,
        algorithm.Status,
        algorithm.Priority,
        ToJsonElement(algorithm.Configuration),
        algorithm.CreatedAt,
        algorithm.UpdatedAt);

    private static PoolDetailsInput ToInput(PoolDetailsRequest details) => new(
        details.BaseUrl,
        details.StatusEndpoint,
        details.StratumEndpoint,
        details.CoinId,
        details.PayoutMode,
        details.PayoutAddress,
        details.PayoutNetwork);

    /// <summary>Free-form JSON is stored verbatim; the wire format is a JSON object.</summary>
    private static string? ToRawJson(JsonElement? element) =>
        element is { ValueKind: not JsonValueKind.Null and not JsonValueKind.Undefined }
            ? element.Value.GetRawText()
            : null;

    private static JsonElement? ToJsonElement(string? rawJson) =>
        string.IsNullOrWhiteSpace(rawJson) ? null : JsonDocument.Parse(rawJson).RootElement.Clone();
}
