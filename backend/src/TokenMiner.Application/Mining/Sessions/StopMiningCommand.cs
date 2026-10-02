using FluentValidation;
using MediatR;
using TokenMiner.Application.Common.Abstractions;
using TokenMiner.Application.Mining.Abstractions;
using TokenMiner.Domain.Mining;
using TokenMiner.Domain.Mining.Enums;

namespace TokenMiner.Application.Mining.Sessions;

public sealed record StopMiningCommand(Guid UserId, Guid HardwareId, string Reason) : IRequest;

public sealed class StopMiningCommandValidator : AbstractValidator<StopMiningCommand>
{
    public StopMiningCommandValidator()
    {
        RuleFor(x => x.UserId).NotEmpty();
        RuleFor(x => x.HardwareId).NotEmpty();
        RuleFor(x => x.Reason)
            .Must(value => MiningStopReasonDbValues.TryParseDbValue(value, out _))
            .WithMessage("Reason must be one of: connection-failure, driver-crash, miner-closed, user-triggered, unknown.");
    }
}

/// <summary>
/// Stops the caller's active session for a device. Idempotent: an unknown device or an
/// already-stopped device is a successful no-op, so a retry never writes a second stop event.
/// </summary>
internal sealed class StopMiningCommandHandler(
    IUserHardwareRepository hardware,
    IMiningSessionRepository sessions,
    IMiningLogRepository logs,
    IUnitOfWork unitOfWork,
    TimeProvider timeProvider) : IRequestHandler<StopMiningCommand>
{
    public async Task Handle(StopMiningCommand request, CancellationToken cancellationToken)
    {
        var device = await hardware.GetByUserAndHardwareIdAsync(
            request.UserId,
            request.HardwareId,
            cancellationToken);

        if (device is null)
        {
            return;
        }

        var session = await sessions.GetActiveByHardwareAsync(device.Id, cancellationToken);
        if (session is null)
        {
            return;
        }

        var now = timeProvider.GetUtcNow();

        if (!session.Stop(request.Reason, now))
        {
            return;
        }

        device.Touch(now);

        logs.Add(new MiningLog(
            Guid.NewGuid(),
            request.UserId,
            device.Id,
            session.Id,
            session.PoolId,
            session.CoinId,
            MiningEventType.MiningStopped,
            reason: request.Reason,
            metadata: null,
            now));

        await unitOfWork.SaveChangesAsync(cancellationToken);
    }
}
