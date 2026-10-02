using FluentValidation;
using MediatR;
using TokenMiner.Application.Common.Abstractions;
using TokenMiner.Application.Common.Exceptions;
using TokenMiner.Application.Mining.Abstractions;
using TokenMiner.Application.Mining.Models;
using TokenMiner.Domain.Mining;
using TokenMiner.Domain.Mining.Enums;

namespace TokenMiner.Application.Mining.Coins;

public sealed record CreateCoinCommand(string Code, string Name, string Network, int Decimals) : IRequest<CoinDto>;

public sealed class CreateCoinCommandValidator : AbstractValidator<CreateCoinCommand>
{
    public CreateCoinCommandValidator()
    {
        RuleFor(x => x.Code)
            .NotEmpty()
            .MaximumLength(32)
            .Matches("^[a-zA-Z0-9-]+$").WithMessage("Code may only contain letters, digits and dashes.");

        RuleFor(x => x.Name).NotEmpty().MaximumLength(128);
        RuleFor(x => x.Network).NotEmpty().MaximumLength(64);
        RuleFor(x => x.Decimals).InclusiveBetween(0, 18);
    }
}

internal sealed class CreateCoinCommandHandler(
    ICoinRepository coins,
    IUnitOfWork unitOfWork,
    TimeProvider timeProvider) : IRequestHandler<CreateCoinCommand, CoinDto>
{
    public async Task<CoinDto> Handle(CreateCoinCommand request, CancellationToken cancellationToken)
    {
        var code = request.Code.Trim().ToLowerInvariant();

        if (await coins.GetByCodeAsync(code, cancellationToken) is not null)
        {
            throw new ConflictException($"A coin with code '{code}' already exists.");
        }

        var coin = new Coin(
            Guid.NewGuid(),
            code,
            request.Name.Trim(),
            request.Network.Trim(),
            request.Decimals,
            timeProvider.GetUtcNow());

        coins.Add(coin);
        await unitOfWork.SaveChangesAsync(cancellationToken);

        return coin.ToDto();
    }
}

public sealed record UpdateCoinCommand(Guid Id, string Name, string Network, int Decimals, string Status)
    : IRequest<CoinDto>;

public sealed class UpdateCoinCommandValidator : AbstractValidator<UpdateCoinCommand>
{
    public UpdateCoinCommandValidator()
    {
        RuleFor(x => x.Id).NotEmpty();
        RuleFor(x => x.Name).NotEmpty().MaximumLength(128);
        RuleFor(x => x.Network).NotEmpty().MaximumLength(64);
        RuleFor(x => x.Decimals).InclusiveBetween(0, 18);
        RuleFor(x => x.Status)
            .Must(value => MiningStatusDbValues.TryParseDbValue(value, out _))
            .WithMessage("Status must be 'active' or 'disabled'.");
    }
}

internal sealed class UpdateCoinCommandHandler(
    ICoinRepository coins,
    IUnitOfWork unitOfWork,
    TimeProvider timeProvider) : IRequestHandler<UpdateCoinCommand, CoinDto>
{
    public async Task<CoinDto> Handle(UpdateCoinCommand request, CancellationToken cancellationToken)
    {
        var coin = await coins.GetByIdAsync(request.Id, cancellationToken)
            ?? throw new NotFoundException("Coin not found.");

        coin.UpdateDetails(
            request.Name.Trim(),
            request.Network.Trim(),
            request.Decimals,
            MiningStatusDbValues.FromDbValue(request.Status),
            timeProvider.GetUtcNow());

        await unitOfWork.SaveChangesAsync(cancellationToken);

        return coin.ToDto();
    }
}
