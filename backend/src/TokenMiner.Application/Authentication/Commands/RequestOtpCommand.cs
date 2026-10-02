using FluentValidation;
using MediatR;
using TokenMiner.Application.Authentication.Abstractions;
using TokenMiner.Application.Common.Abstractions;
using TokenMiner.Application.Common.Exceptions;
using TokenMiner.Domain.Users;
using TokenMiner.Domain.Users.Enums;

namespace TokenMiner.Application.Authentication.Commands;

public sealed record RequestOtpCommand(string Email) : IRequest;

public sealed class RequestOtpCommandValidator : AbstractValidator<RequestOtpCommand>
{
    public RequestOtpCommandValidator()
    {
        RuleFor(x => x.Email).NotEmpty().EmailAddress().MaximumLength(320);
    }
}

/// <summary>
/// Resends an activation code. Responds identically whether or not the account exists,
/// so the endpoint cannot be used to enumerate registered emails.
/// </summary>
internal sealed class RequestOtpCommandHandler(
    IUserRepository users,
    IOtpStore otpStore,
    IOtpService otpService,
    IEmailSender emailSender,
    IUnitOfWork unitOfWork,
    TimeProvider timeProvider,
    AuthOptions options) : IRequestHandler<RequestOtpCommand>
{
    public async Task Handle(RequestOtpCommand request, CancellationToken cancellationToken)
    {
        var user = await users.GetByNormalizedEmailAsync(
            EmailNormalizer.Normalize(request.Email),
            cancellationToken);

        if (user is null)
        {
            return;
        }

        var now = timeProvider.GetUtcNow();

        var lastCreatedAt = await otpStore.GetLastCreatedAtAsync(user.Id, OtpPurpose.Activation, cancellationToken);
        if (lastCreatedAt is not null
            && now - lastCreatedAt.Value < TimeSpan.FromSeconds(options.Otp.ResendCooldownSeconds))
        {
            throw new TooManyRequestsException("Please wait before requesting another code.");
        }

        await otpStore.InvalidateOutstandingAsync(user.Id, OtpPurpose.Activation, now, cancellationToken);

        var code = otpService.GenerateCode();
        otpStore.Add(new OtpCode(
            Guid.NewGuid(),
            user.Id,
            OtpPurpose.Activation,
            otpService.Hash(code),
            now.AddMinutes(options.Otp.ExpiryMinutes),
            options.Otp.MaxAttempts,
            now));

        await unitOfWork.SaveChangesAsync(cancellationToken);
        await emailSender.SendActivationOtpAsync(user.Email, code, cancellationToken);
    }
}
