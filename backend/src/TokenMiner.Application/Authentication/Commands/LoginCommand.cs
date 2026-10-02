using FluentValidation;
using MediatR;
using TokenMiner.Application.Authentication.Abstractions;
using TokenMiner.Application.Authentication.Models;
using TokenMiner.Application.Common.Abstractions;
using TokenMiner.Application.Common.Exceptions;
using TokenMiner.Domain.Users;

namespace TokenMiner.Application.Authentication.Commands;

public sealed record LoginCommand(string Email, string Password, string? Ip, string? UserAgent)
    : IRequest<AuthTokens>;

public sealed class LoginCommandValidator : AbstractValidator<LoginCommand>
{
    public LoginCommandValidator()
    {
        RuleFor(x => x.Email).NotEmpty().EmailAddress().MaximumLength(320);
        RuleFor(x => x.Password).NotEmpty().MaximumLength(128);
    }
}

internal sealed class LoginCommandHandler(
    IUserRepository users,
    ILoginAttemptStore loginAttempts,
    IPasswordHasher passwordHasher,
    IAuthTokenIssuer tokenIssuer,
    IUnitOfWork unitOfWork,
    TimeProvider timeProvider,
    AuthOptions options) : IRequestHandler<LoginCommand, AuthTokens>
{
    public async Task<AuthTokens> Handle(LoginCommand request, CancellationToken cancellationToken)
    {
        var normalizedEmail = EmailNormalizer.Normalize(request.Email);
        var now = timeProvider.GetUtcNow();

        var windowStart = now.AddMinutes(-options.Lockout.WindowMinutes);
        var recentFailures = await loginAttempts.CountRecentFailuresAsync(normalizedEmail, windowStart, cancellationToken);
        if (recentFailures >= options.Lockout.MaxFailedAttempts)
        {
            throw new TooManyRequestsException("Too many failed sign-in attempts. Try again later.");
        }

        var user = await users.GetByNormalizedEmailAsync(normalizedEmail, cancellationToken);

        // Unknown accounts and accounts without a password both fall through to the same
        // generic failure, so the response reveals nothing about which emails exist.
        if (user?.PasswordHash is null || !passwordHasher.Verify(request.Password, user.PasswordHash))
        {
            loginAttempts.Add(new LoginAttempt(Guid.NewGuid(), normalizedEmail, request.Ip, succeeded: false, now));
            await unitOfWork.SaveChangesAsync(cancellationToken);

            throw new UnauthorizedException("Invalid email or password.");
        }

        if (!user.IsActive)
        {
            throw new ForbiddenException("Account is not activated.");
        }

        loginAttempts.Add(new LoginAttempt(Guid.NewGuid(), normalizedEmail, request.Ip, succeeded: true, now));

        var roles = await users.GetRoleNamesAsync(user.Id, cancellationToken);
        var issued = await tokenIssuer.IssueAsync(user, roles, request.Ip, request.UserAgent, cancellationToken);

        await unitOfWork.SaveChangesAsync(cancellationToken);

        return issued.Tokens;
    }
}
