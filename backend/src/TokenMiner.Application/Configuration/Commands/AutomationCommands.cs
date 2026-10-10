using MediatR;
using TokenMiner.Application.Common.Abstractions;
using TokenMiner.Application.Configuration.Abstractions;
using TokenMiner.Application.Mining;
using TokenMiner.Application.Mining.Abstractions;
using TokenMiner.Application.Mining.Services;
using TokenMiner.Application.Providers;
using TokenMiner.Application.Providers.Abstractions;
using TokenMiner.Application.Providers.Commands;
using TokenMiner.Application.Treasury.Abstractions;
using TokenMiner.Application.Treasury.Commands;
using TokenMiner.Domain.Configuration;
using TokenMiner.Domain.Mining.Enums;

namespace TokenMiner.Application.Configuration.Commands;

public enum TopUpOutcome
{
    /// <summary>The provider already holds enough credit.</summary>
    Skipped = 0,

    /// <summary>Stages were driven towards a top-up.</summary>
    Driven = 1,

    /// <summary>Auto top-up is switched off.</summary>
    Disabled = 2,
}

public sealed record TopUpStatus(
    TopUpOutcome Outcome,
    decimal? Balance,
    decimal Target,
    decimal Minimum,
    int PayoutsRecorded,
    int SharesRedeemed,
    int ConversionsQueued,
    int ConversionsAdvanced,
    int DepositsQueued,
    int DepositsAdvanced);

public sealed record RunAutoTopUpCommand : IRequest<TopUpStatus>;

/// <summary>
/// Drives the treasury towards a topped-up provider balance.
/// </summary>
/// <remarks>
/// This is the state machine's tick, not a monolith: each stage is its own idempotent command,
/// so running this repeatedly is safe and each stage can also be run on its own. The decision to
/// act at all is the only thing this command owns.
/// </remarks>
internal sealed class RunAutoTopUpCommandHandler(
    ISystemConfigurationStore configuration,
    ILlmProviderRepository providers,
    IProviderDepositRepository deposits,
    ISender sender,
    ProviderOptions providerOptions)
    : IRequestHandler<RunAutoTopUpCommand, TopUpStatus>
{
    public async Task<TopUpStatus> Handle(RunAutoTopUpCommand request, CancellationToken cancellationToken)
    {
        var target = await configuration.GetDecimalAsync(
            SystemConfigurationKeys.ProviderBalanceTarget,
            providerOptions.BalanceTarget,
            cancellationToken);

        var minimum = await configuration.GetDecimalAsync(
            SystemConfigurationKeys.ProviderBalanceMinimum,
            providerOptions.BalanceMinimum,
            cancellationToken);

        var enabled = await configuration.GetBoolAsync(
            SystemConfigurationKeys.AutoTopUpEnabled,
            true,
            cancellationToken);

        var active = (await providers.ListProvidersAsync(cancellationToken))
            .FirstOrDefault(provider => provider.Status == Domain.Mining.Enums.MiningStatus.Active);

        if (active is null)
        {
            return Idle(TopUpOutcome.Skipped, null, target, minimum);
        }

        var balance = (await providers.GetLatestBalanceAsync(active.Id, cancellationToken))?.Balance;

        if (!enabled)
        {
            return Idle(TopUpOutcome.Disabled, balance, target, minimum);
        }

        // Enough credit already: nothing to do, and nothing on chain to wait for.
        var inFlight = (await deposits.ListInFlightAsync(cancellationToken))
            .Count(deposit => deposit.LlmProviderId == active.Id);

        if (balance is not null && balance.Value >= minimum && inFlight == 0)
        {
            return Idle(TopUpOutcome.Skipped, balance, target, minimum);
        }

        // Each stage is idempotent, so the pipeline can simply be advanced.
        var monitor = await sender.Send(new MonitorPoolsCommand(), cancellationToken);
        var reconciliation = await sender.Send(new ReconcilePayoutsCommand(), cancellationToken);
        var conversion = await sender.Send(new RunConversionPipelineCommand(providerOptions.DepositBatchSize), cancellationToken);
        var deposit = await sender.Send(new RunProviderDepositPipelineCommand(), cancellationToken);

        return new TopUpStatus(
            TopUpOutcome.Driven,
            balance,
            target,
            minimum,
            monitor.PayoutsRecorded,
            reconciliation.SharesRedeemed,
            conversion.Queued,
            conversion.Advanced,
            deposit.DepositsQueued,
            deposit.DepositsAdvanced);
    }

    private static TopUpStatus Idle(TopUpOutcome outcome, decimal? balance, decimal target, decimal minimum) =>
        new(outcome, balance, target, minimum, 0, 0, 0, 0, 0, 0);
}

public sealed record SetSystemConfigurationCommand(string Key, string Value) : IRequest<SystemConfigurationDto>;

public sealed record SystemConfigurationDto(string Key, string Value, DateTimeOffset UpdatedAt);

internal sealed class SetSystemConfigurationCommandHandler(
    ISystemConfigurationStore configuration,
    TimeProvider timeProvider) : IRequestHandler<SetSystemConfigurationCommand, SystemConfigurationDto>
{
    public async Task<SystemConfigurationDto> Handle(
        SetSystemConfigurationCommand request,
        CancellationToken cancellationToken)
    {
        var now = timeProvider.GetUtcNow();

        // The store commits the setting itself, so there is no unit of work to flush here.
        await configuration.SetAsync(request.Key, request.Value, now, cancellationToken);

        return new SystemConfigurationDto(request.Key, request.Value, now);
    }
}

public sealed record ListSystemConfigurationsQuery : IRequest<IReadOnlyList<SystemConfigurationDto>>;

internal sealed class ListSystemConfigurationsQueryHandler(ISystemConfigurationStore configuration)
    : IRequestHandler<ListSystemConfigurationsQuery, IReadOnlyList<SystemConfigurationDto>>
{
    public async Task<IReadOnlyList<SystemConfigurationDto>> Handle(
        ListSystemConfigurationsQuery request,
        CancellationToken cancellationToken) =>
        (await configuration.ListAsync(cancellationToken))
            .OrderBy(setting => setting.Key, StringComparer.Ordinal)
            .Select(setting => new SystemConfigurationDto(setting.Key, setting.Value, setting.UpdatedAt))
            .ToList();
}

/// <summary>An operational snapshot: what is in flight and what needs attention.</summary>
public sealed record SystemStatusDto(
    int RunningSessions,
    int IdleSessions,
    int PendingShares,
    int ConversionsInFlight,
    int ProviderDepositsInFlight,
    decimal? ProviderBalance,
    decimal? ProviderReservedBalance,
    DateTimeOffset? ProviderBalanceCheckedAt,
    DateTimeOffset CapturedAt);

public sealed record GetSystemStatusQuery : IRequest<SystemStatusDto>;

internal sealed class GetSystemStatusQueryHandler(
    IMiningSessionRepository sessions,
    IShareRepository shares,
    IConversionRepository conversions,
    IProviderDepositRepository deposits,
    ILlmProviderRepository providers,
    MiningOptions miningOptions,
    TimeProvider timeProvider) : IRequestHandler<GetSystemStatusQuery, SystemStatusDto>
{
    public async Task<SystemStatusDto> Handle(
        GetSystemStatusQuery request,
        CancellationToken cancellationToken)
    {
        var now = timeProvider.GetUtcNow();

        var activeProvider = (await providers.ListProvidersAsync(cancellationToken)).FirstOrDefault();
        var balance = activeProvider is null
            ? null
            : await providers.GetLatestBalanceAsync(activeProvider.Id, cancellationToken);

        // Status is a projection over each device's last accepted share; no status is stored.
        var activeSessions = await sessions.ListActiveAsync(cancellationToken);
        var lastShareByDevice = await shares.GetLastShareAtByHardwareAsync(
            activeSessions.Select(session => session.UserHardwareId),
            cancellationToken);

        var running = activeSessions.Count(session =>
            MiningStatusProjection.Resolve(
                lastShareByDevice.TryGetValue(session.UserHardwareId, out var last) ? last : null,
                now,
                miningOptions.ShareActivityWindowSeconds) == MiningStatusProjection.Running);

        return new SystemStatusDto(
            running,
            activeSessions.Count - running,
            await shares.CountByStatusAsync(ShareRewardStatus.Pending, cancellationToken),
            await conversions.CountInFlightAsync(cancellationToken),
            await deposits.CountInFlightAsync(cancellationToken),
            balance?.Balance,
            balance?.ReservedBalance,
            balance?.CheckedAt,
            now);
    }
}
