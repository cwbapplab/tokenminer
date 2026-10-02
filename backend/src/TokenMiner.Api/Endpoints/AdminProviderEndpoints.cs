using MediatR;
using TokenMiner.Application.Providers.Models;
using TokenMiner.Contracts.Providers;

namespace TokenMiner.Api.Endpoints;

/// <summary>
/// Operator surface for the LLM provider: configuration, deposit rails, the synced model
/// catalogue, balances and the deposit ledger.
/// </summary>
public static class AdminProviderEndpoints
{
    public static IEndpointRouteBuilder MapAdminProviderEndpoints(this IEndpointRouteBuilder app)
    {
        ArgumentNullException.ThrowIfNull(app);

        var group = app.MapGroup("/api/admin/llm-providers")
            .WithTags("Admin - Providers")
            .RequireAuthorization(AuthorizationPolicies.Admin);

        group.MapGet(string.Empty, async (ISender sender, CancellationToken cancellationToken) =>
        {
            var providers = await sender.Send(new ListLlmProvidersQuery(), cancellationToken);
            return Results.Ok(providers.Select(ToResponse).ToList());
        })
        .WithName("AdminListLlmProviders")
        .Produces<IReadOnlyList<LlmProviderResponse>>();

        group.MapPost(string.Empty, async (
            CreateLlmProviderRequest request,
            ISender sender,
            CancellationToken cancellationToken) =>
        {
            var provider = await sender.Send(
                new CreateLlmProviderCommand(request.Name, request.Endpoint, request.CredentialRef),
                cancellationToken);

            return Results.Created($"/api/admin/llm-providers/{provider.Id}", ToResponse(provider));
        })
        .WithName("AdminCreateLlmProvider")
        .Produces<LlmProviderResponse>(StatusCodes.Status201Created)
        .ProducesProblem(StatusCodes.Status409Conflict);

        group.MapPut("/{id:guid}", async (
            Guid id,
            UpdateLlmProviderRequest request,
            ISender sender,
            CancellationToken cancellationToken) =>
        {
            var provider = await sender.Send(
                new UpdateLlmProviderCommand(id, request.Name, request.Endpoint, request.CredentialRef, request.Status),
                cancellationToken);

            return Results.Ok(ToResponse(provider));
        })
        .WithName("AdminUpdateLlmProvider")
        .Produces<LlmProviderResponse>()
        .ProducesProblem(StatusCodes.Status404NotFound);

        group.MapPut("/{id:guid}/deposit-accounts", async (
            Guid id,
            UpsertDepositAccountRequest request,
            ISender sender,
            CancellationToken cancellationToken) =>
        {
            var account = await sender.Send(
                new UpsertDepositAccountCommand(id, request.CoinId, request.Network, request.DepositAddress, request.Status),
                cancellationToken);

            return Results.Ok(ToResponse(account));
        })
        .WithName("AdminUpsertDepositAccount")
        .Produces<LlmProviderDepositAccountResponse>()
        .ProducesProblem(StatusCodes.Status404NotFound);

        group.MapGet("/{id:guid}/models", async (
            Guid id,
            ISender sender,
            CancellationToken cancellationToken) =>
        {
            var models = await sender.Send(new ListProviderModelsQuery(id), cancellationToken);
            return Results.Ok(models.Select(ToResponse).ToList());
        })
        .WithName("AdminListProviderModels")
        .Produces<IReadOnlyList<LlmProviderModelResponse>>();

        app.MapGet("/api/admin/provider-deposits", async (
            int? limit,
            ISender sender,
            CancellationToken cancellationToken) =>
        {
            var deposits = await sender.Send(new ListProviderDepositsQuery(limit ?? 100), cancellationToken);
            return Results.Ok(deposits.Select(ToResponse).ToList());
        })
        .WithTags("Admin - Providers")
        .WithName("AdminListProviderDeposits")
        .RequireAuthorization(AuthorizationPolicies.Admin)
        .Produces<IReadOnlyList<ProviderDepositResponse>>();

        return app;
    }

    private static LlmProviderResponse ToResponse(LlmProviderDto provider) => new(
        provider.Id,
        provider.Name,
        provider.Endpoint,
        provider.CredentialRef,
        provider.Status,
        provider.Balance,
        provider.ReservedBalance,
        provider.TotalBalance,
        provider.BalanceCheckedAt,
        provider.DepositAccounts.Select(ToResponse).ToList(),
        provider.CreatedAt,
        provider.UpdatedAt);

    private static LlmProviderDepositAccountResponse ToResponse(LlmProviderDepositAccountDto account) => new(
        account.Id,
        account.CoinId,
        account.CoinCode,
        account.Network,
        account.DepositAddress,
        account.Status);

    private static LlmProviderModelResponse ToResponse(LlmProviderModelDto model) => new(
        model.Id,
        model.ModelId,
        model.Name,
        model.InputCost,
        model.OutputCost,
        model.CachedInputCost,
        model.Currency,
        model.ContextLength,
        model.Capabilities,
        model.Status,
        model.LastSyncedAt);

    private static ProviderDepositResponse ToResponse(ProviderDepositDto deposit) => new(
        deposit.Id,
        deposit.LlmProviderId,
        deposit.CoinId,
        deposit.Network,
        deposit.Address,
        deposit.Amount,
        deposit.TransactionHash,
        deposit.Confirmations,
        deposit.Status,
        deposit.ProviderCreditBefore,
        deposit.ProviderCreditAfter,
        deposit.IdempotencyKey,
        deposit.Error,
        deposit.CreatedAt,
        deposit.UpdatedAt,
        deposit.ConfirmedAt);
}
