using System.Text.Json;
using FluentValidation;
using MediatR;
using TokenMiner.Application.Common.Abstractions;
using TokenMiner.Application.Common.Exceptions;
using TokenMiner.Application.Mining.Abstractions;
using TokenMiner.Application.Mining.Models;
using TokenMiner.Domain.Mining;
using TokenMiner.Domain.Mining.Enums;

namespace TokenMiner.Application.Mining.Algorithms;

public sealed record CreateMiningAlgoCommand(
    string Code,
    string Name,
    int Priority,
    string? Configuration) : IRequest<MiningAlgoDto>;

public sealed class CreateMiningAlgoCommandValidator : AbstractValidator<CreateMiningAlgoCommand>
{
    public CreateMiningAlgoCommandValidator()
    {
        RuleFor(x => x.Code)
            .NotEmpty()
            .MaximumLength(64)
            .Matches("^[a-zA-Z0-9._-]+$").WithMessage("Code may only contain letters, digits, dots, underscores and dashes.");

        RuleFor(x => x.Name).NotEmpty().MaximumLength(128);
        RuleFor(x => x.Priority).GreaterThanOrEqualTo(0);

        RuleFor(x => x.Configuration)
            .Must(BeValidJson)
            .When(x => !string.IsNullOrWhiteSpace(x.Configuration))
            .WithMessage("Configuration must be valid JSON.");
    }

    private static bool BeValidJson(string? value)
    {
        try
        {
            using var _ = JsonDocument.Parse(value!);
            return true;
        }
        catch (JsonException)
        {
            return false;
        }
    }
}

internal sealed class CreateMiningAlgoCommandHandler(
    IMiningAlgoRepository algorithms,
    IUnitOfWork unitOfWork,
    TimeProvider timeProvider) : IRequestHandler<CreateMiningAlgoCommand, MiningAlgoDto>
{
    public async Task<MiningAlgoDto> Handle(CreateMiningAlgoCommand request, CancellationToken cancellationToken)
    {
        var code = request.Code.Trim().ToLowerInvariant();

        if (await algorithms.GetByCodeAsync(code, cancellationToken) is not null)
        {
            throw new ConflictException($"A mining algorithm with code '{code}' already exists.");
        }

        var algorithm = new MiningAlgo(
            Guid.NewGuid(),
            code,
            request.Name.Trim(),
            request.Priority,
            request.Configuration,
            timeProvider.GetUtcNow());

        algorithms.Add(algorithm);
        await unitOfWork.SaveChangesAsync(cancellationToken);

        return algorithm.ToDto();
    }
}

public sealed record UpdateMiningAlgoCommand(
    Guid Id,
    string Name,
    int Priority,
    string Status,
    string? Configuration) : IRequest<MiningAlgoDto>;

public sealed class UpdateMiningAlgoCommandValidator : AbstractValidator<UpdateMiningAlgoCommand>
{
    public UpdateMiningAlgoCommandValidator()
    {
        RuleFor(x => x.Id).NotEmpty();
        RuleFor(x => x.Name).NotEmpty().MaximumLength(128);
        RuleFor(x => x.Priority).GreaterThanOrEqualTo(0);
        RuleFor(x => x.Status)
            .Must(value => MiningStatusDbValues.TryParseDbValue(value, out _))
            .WithMessage("Status must be 'active' or 'disabled'.");

        RuleFor(x => x.Configuration)
            .Must(BeValidJson)
            .When(x => !string.IsNullOrWhiteSpace(x.Configuration))
            .WithMessage("Configuration must be valid JSON.");
    }

    private static bool BeValidJson(string? value)
    {
        try
        {
            using var _ = JsonDocument.Parse(value!);
            return true;
        }
        catch (JsonException)
        {
            return false;
        }
    }
}

internal sealed class UpdateMiningAlgoCommandHandler(
    IMiningAlgoRepository algorithms,
    IUnitOfWork unitOfWork,
    TimeProvider timeProvider) : IRequestHandler<UpdateMiningAlgoCommand, MiningAlgoDto>
{
    public async Task<MiningAlgoDto> Handle(UpdateMiningAlgoCommand request, CancellationToken cancellationToken)
    {
        var algorithm = await algorithms.GetByIdAsync(request.Id, cancellationToken)
            ?? throw new NotFoundException("Mining algorithm not found.");

        algorithm.Update(
            request.Name.Trim(),
            request.Priority,
            request.Configuration,
            MiningStatusDbValues.FromDbValue(request.Status),
            timeProvider.GetUtcNow());

        await unitOfWork.SaveChangesAsync(cancellationToken);

        return algorithm.ToDto();
    }
}
