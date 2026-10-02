using MediatR;
using TokenMiner.Application.Treasury.Abstractions;
using TokenMiner.Application.Treasury.Commands;
using TokenMiner.Application.Treasury.Models;
using TokenMiner.Application.Treasury.Queries;
using TokenMiner.Contracts.Treasury;

namespace TokenMiner.Api.Endpoints;

/// <summary>
/// Operator surface for the treasury: conversion configuration, the conversion ledger, and the
/// manual settlement that the manual provider depends on.
/// </summary>
public static class AdminTreasuryEndpoints
{
    public static IEndpointRouteBuilder MapAdminTreasuryEndpoints(this IEndpointRouteBuilder app)
    {
        ArgumentNullException.ThrowIfNull(app);

        MapConversionProviders(app);
        MapConversionRoutes(app);
        MapConversions(app);
        MapPayouts(app);

        return app;
    }

    private static void MapConversionProviders(IEndpointRouteBuilder app)
    {
        var group = app.MapGroup("/api/admin/conversion-providers")
            .WithTags("Admin - Treasury")
            .RequireAuthorization(AuthorizationPolicies.Admin);

        group.MapGet(string.Empty, async (ISender sender, CancellationToken cancellationToken) =>
        {
            var providers = await sender.Send(new ListConversionProvidersQuery(), cancellationToken);
            return Results.Ok(providers.Select(ToResponse).ToList());
        })
        .WithName("AdminListConversionProviders")
        .Produces<IReadOnlyList<ConversionProviderResponse>>();

        group.MapPost(string.Empty, async (
            CreateConversionProviderRequest request,
            ISender sender,
            CancellationToken cancellationToken) =>
        {
            var provider = await sender.Send(
                new CreateConversionProviderCommand(
                    request.Name,
                    request.BaseUrl,
                    request.Priority,
                    request.CredentialRef,
                    request.SupportedFeatures),
                cancellationToken);

            return Results.Created($"/api/admin/conversion-providers/{provider.Id}", ToResponse(provider));
        })
        .WithName("AdminCreateConversionProvider")
        .Produces<ConversionProviderResponse>(StatusCodes.Status201Created)
        .ProducesProblem(StatusCodes.Status409Conflict);

        group.MapPut("/{id:guid}", async (
            Guid id,
            UpdateConversionProviderRequest request,
            ISender sender,
            CancellationToken cancellationToken) =>
        {
            var provider = await sender.Send(
                new UpdateConversionProviderCommand(
                    id,
                    request.BaseUrl,
                    request.Priority,
                    request.Status,
                    request.CredentialRef,
                    request.SupportedFeatures),
                cancellationToken);

            return Results.Ok(ToResponse(provider));
        })
        .WithName("AdminUpdateConversionProvider")
        .Produces<ConversionProviderResponse>()
        .ProducesProblem(StatusCodes.Status404NotFound);
    }

    private static void MapConversionRoutes(IEndpointRouteBuilder app)
    {
        var group = app.MapGroup("/api/admin/conversion-routes")
            .WithTags("Admin - Treasury")
            .RequireAuthorization(AuthorizationPolicies.Admin);

        group.MapGet(string.Empty, async (ISender sender, CancellationToken cancellationToken) =>
        {
            var routes = await sender.Send(new ListConversionRoutesQuery(), cancellationToken);
            return Results.Ok(routes.Select(ToResponse).ToList());
        })
        .WithName("AdminListConversionRoutes")
        .Produces<IReadOnlyList<ConversionRouteResponse>>();

        group.MapPost(string.Empty, async (
            CreateConversionRouteRequest request,
            ISender sender,
            CancellationToken cancellationToken) =>
        {
            var route = await sender.Send(
                new CreateConversionRouteCommand(
                    request.ConversionProviderId,
                    request.SourceCoinId,
                    request.DestinationCoinId,
                    request.SourceNetwork,
                    request.DestinationNetwork,
                    request.Priority,
                    request.MinimumAmount),
                cancellationToken);

            return Results.Created($"/api/admin/conversion-routes/{route.Id}", ToResponse(route));
        })
        .WithName("AdminCreateConversionRoute")
        .Produces<ConversionRouteResponse>(StatusCodes.Status201Created)
        .ProducesProblem(StatusCodes.Status404NotFound);

        group.MapPut("/{id:guid}", async (
            Guid id,
            UpdateConversionRouteRequest request,
            ISender sender,
            CancellationToken cancellationToken) =>
        {
            var route = await sender.Send(
                new UpdateConversionRouteCommand(id, request.Enabled, request.Priority, request.MinimumAmount),
                cancellationToken);

            return Results.Ok(ToResponse(route));
        })
        .WithName("AdminUpdateConversionRoute")
        .Produces<ConversionRouteResponse>()
        .ProducesProblem(StatusCodes.Status404NotFound);
    }

    private static void MapConversions(IEndpointRouteBuilder app)
    {
        var group = app.MapGroup("/api/admin/conversions")
            .WithTags("Admin - Treasury")
            .RequireAuthorization(AuthorizationPolicies.Admin);

        group.MapGet(string.Empty, async (int? limit, ISender sender, CancellationToken cancellationToken) =>
        {
            var transactions = await sender.Send(
                new ListConversionsQuery(limit ?? 100),
                cancellationToken);

            return Results.Ok(transactions.Select(ToResponse).ToList());
        })
        .WithName("AdminListConversions")
        .Produces<IReadOnlyList<ConversionTransactionResponse>>();

        group.MapPost("/{id:guid}/settlement", async (
            Guid id,
            SettleConversionRequest request,
            ISender sender,
            CancellationToken cancellationToken) =>
        {
            if (!Enum.TryParse<ConversionSettlementOutcome>(request.Outcome, ignoreCase: true, out var outcome))
            {
                return Results.Problem(
                    detail: "Outcome must be 'completed', 'failed' or 'cancelled'.",
                    statusCode: StatusCodes.Status400BadRequest);
            }

            var settled = await sender.Send(
                new SettleConversionCommand(
                    id,
                    outcome,
                    request.DestinationAmount,
                    request.ExchangeRate,
                    request.Fees,
                    request.DestinationTransactionId,
                    request.Reason),
                cancellationToken);

            return settled
                ? Results.NoContent()
                : Results.Problem(detail: "Conversion not found.", statusCode: StatusCodes.Status404NotFound);
        })
        .WithName("AdminSettleConversion")
        .Produces(StatusCodes.Status204NoContent)
        .ProducesProblem(StatusCodes.Status400BadRequest)
        .ProducesProblem(StatusCodes.Status404NotFound);
    }

    private static void MapPayouts(IEndpointRouteBuilder app)
    {
        app.MapGet("/api/admin/pool-payouts", async (
            int? limit,
            ISender sender,
            CancellationToken cancellationToken) =>
        {
            var payouts = await sender.Send(new ListPoolPayoutsQuery(limit ?? 100), cancellationToken);
            return Results.Ok(payouts.Select(ToResponse).ToList());
        })
        .WithTags("Admin - Treasury")
        .WithName("AdminListPoolPayouts")
        .RequireAuthorization(AuthorizationPolicies.Admin)
        .Produces<IReadOnlyList<PoolPayoutResponse>>();
    }

    // --- Mapping -------------------------------------------------------------------------

    private static ConversionProviderResponse ToResponse(ConversionProviderDto provider) => new(
        provider.Id,
        provider.Name,
        provider.BaseUrl,
        provider.Status,
        provider.Priority,
        provider.CredentialRef,
        provider.SupportedFeatures,
        provider.CreatedAt,
        provider.UpdatedAt);

    private static ConversionRouteResponse ToResponse(ConversionRouteDto route) => new(
        route.Id,
        route.ConversionProviderId,
        route.ConversionProviderName,
        route.SourceCoinId,
        route.SourceCoinCode,
        route.DestinationCoinId,
        route.DestinationCoinCode,
        route.SourceNetwork,
        route.DestinationNetwork,
        route.Enabled,
        route.Priority,
        route.MinimumAmount,
        route.CreatedAt,
        route.UpdatedAt);

    private static ConversionTransactionResponse ToResponse(ConversionTransactionDto transaction) => new(
        transaction.Id,
        transaction.ConversionProviderId,
        transaction.ConversionProviderName,
        transaction.SourceCoinCode,
        transaction.SourceAmount,
        transaction.DestinationCoinCode,
        transaction.DestinationAmount,
        transaction.ExchangeRate,
        transaction.Fees,
        transaction.SourceTransactionId,
        transaction.DestinationTransactionId,
        transaction.Status,
        transaction.IdempotencyKey,
        transaction.Error,
        transaction.CreatedAt,
        transaction.UpdatedAt,
        transaction.CompletedAt);

    private static PoolPayoutResponse ToResponse(PoolPayoutDto payout) => new(
        payout.Id,
        payout.PoolId,
        payout.CoinId,
        payout.WalletAddress,
        payout.Amount,
        payout.RedeemedAmount,
        payout.TransactionHash,
        payout.RequestedAt,
        payout.ReceivedAt,
        payout.ConfirmationsCount,
        payout.Status,
        payout.CreatedAt,
        payout.UpdatedAt);
}
