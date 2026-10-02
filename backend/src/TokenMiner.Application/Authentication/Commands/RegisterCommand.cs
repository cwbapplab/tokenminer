using FluentValidation;
using MediatR;
using TokenMiner.Application.Authentication.Abstractions;
using TokenMiner.Application.Common.Abstractions;
using TokenMiner.Application.Common.Exceptions;
using TokenMiner.Domain.Users;
using TokenMiner.Domain.Users.Enums;

namespace TokenMiner.Application.Authentication.Commands;

public sealed record RegisterCommand(string Email, string Password, string? DisplayName) : IRequest<RegisterResult>;

public sealed record RegisterResult(bool RequiresActivation);

public sealed class RegisterCommandValidator : AbstractValidator<RegisterCommand>
{
    public RegisterCommandValidator()
    {
        RuleFor(x => x.Email)
            .NotEmpty()
            .EmailAddress()
            .MaximumLength(320);

        RuleFor(x => x.Password)
            .NotEmpty()
            .MinimumLength(12)
            .MaximumLength(128)
            .Matches("[A-Za-z]").WithMessage("Password must contain at least one letter.")
            .Matches("[0-9]").WithMessage("Password must contain at least one digit.");

        RuleFor(x => x.DisplayName).MaximumLength(128);
    }
}

internal sealed class RegisterCommandHandler(
    IUserRepository users,
    IOtpStore otpStore,
    IEmailVerificationTokenStore emailVerificationTokens,
    IOtpService otpService,
    ISecureTokenService secureTokenService,
    IPasswordHasher passwordHasher,
    IEmailSender emailSender,
    IUnitOfWork unitOfWork,
    TimeProvider timeProvider,
    AuthOptions options) : IRequestHandler<RegisterCommand, RegisterResult>
{
    public async Task<RegisterResult> Handle(RegisterCommand request, CancellationToken cancellationToken)
    {
        var normalizedEmail = EmailNormalizer.Normalize(request.Email);

        if (await users.ExistsByNormalizedEmailAsync(normalizedEmail, cancellationToken))
        {
            throw new ConflictException("An account with this email already exists.");
        }

        var now = timeProvider.GetUtcNow();

        var user = new User(
            Guid.NewGuid(),
            request.Email.Trim(),
            normalizedEmail,
            passwordHasher.Hash(request.Password),
            request.DisplayName?.Trim(),
            now);

        var defaultRole = await users.GetRoleByNameAsync(Role.User, cancellationToken)
            ?? throw new InvalidOperationException($"Default role '{Role.User}' is not seeded.");

        users.Add(user);
        users.AddUserRole(new UserRole(user.Id, defaultRole.Id));

        var verificationToken = secureTokenService.GenerateToken();
        emailVerificationTokens.Add(new EmailVerificationToken(
            Guid.NewGuid(),
            user.Id,
            secureTokenService.Hash(verificationToken),
            now.AddHours(options.Otp.EmailVerificationTokenLifetimeHours),
            now));

        var otpCode = otpService.GenerateCode();
        otpStore.Add(new OtpCode(
            Guid.NewGuid(),
            user.Id,
            OtpPurpose.Activation,
            otpService.Hash(otpCode),
            now.AddMinutes(options.Otp.ExpiryMinutes),
            options.Otp.MaxAttempts,
            now));

        await unitOfWork.SaveChangesAsync(cancellationToken);

        // Only sent once the account is durably stored, so a failed write cannot email a dead link.
        await emailSender.SendEmailVerificationAsync(user.Email, verificationToken, cancellationToken);
        await emailSender.SendActivationOtpAsync(user.Email, otpCode, cancellationToken);

        return new RegisterResult(RequiresActivation: true);
    }
}
