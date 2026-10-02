using FluentValidation;
using MediatR;
using TokenMiner.Application.Authentication.Abstractions;
using TokenMiner.Application.Common.Abstractions;
using TokenMiner.Application.Common.Exceptions;

namespace TokenMiner.Application.Authentication.Commands;

public sealed record VerifyEmailCommand(string Token) : IRequest;

public sealed class VerifyEmailCommandValidator : AbstractValidator<VerifyEmailCommand>
{
    public VerifyEmailCommandValidator()
    {
        RuleFor(x => x.Token).NotEmpty();
    }
}

internal sealed class VerifyEmailCommandHandler(
    IUserRepository users,
    IEmailVerificationTokenStore emailVerificationTokens,
    ISecureTokenService secureTokenService,
    IUnitOfWork unitOfWork,
    TimeProvider timeProvider) : IRequestHandler<VerifyEmailCommand>
{
    public async Task Handle(VerifyEmailCommand request, CancellationToken cancellationToken)
    {
        var now = timeProvider.GetUtcNow();

        var token = await emailVerificationTokens.GetByHashAsync(
            secureTokenService.Hash(request.Token),
            cancellationToken);

        if (token is null || !token.IsUsable(now))
        {
            throw new UnauthorizedException("Invalid or expired verification token.");
        }

        var user = await users.GetByIdAsync(token.UserId, cancellationToken)
            ?? throw new UnauthorizedException("Invalid or expired verification token.");

        token.MarkUsed(now);
        user.ConfirmEmail(now);

        await unitOfWork.SaveChangesAsync(cancellationToken);
    }
}
