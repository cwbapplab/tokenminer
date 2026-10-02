using MailKit.Net.Smtp;
using MailKit.Security;
using MimeKit;
using TokenMiner.Application.Authentication;
using TokenMiner.Application.Authentication.Abstractions;

namespace TokenMiner.Infrastructure.Authentication;

internal sealed class SmtpEmailSender(SmtpOptions smtpOptions, AuthOptions authOptions) : IEmailSender
{
    public Task SendEmailVerificationAsync(string email, string token, CancellationToken cancellationToken)
    {
        var verificationUrl = authOptions.EmailVerificationUrlTemplate
            .Replace("{token}", Uri.EscapeDataString(token), StringComparison.Ordinal);

        var body = $"""
            <p>Confirm your TokenMiner email address by opening this link:</p>
            <p><a href="{verificationUrl}">{verificationUrl}</a></p>
            <p>If you did not create an account, you can ignore this message.</p>
            """;

        return SendAsync(email, "Confirm your TokenMiner email address", body, cancellationToken);
    }

    public Task SendActivationOtpAsync(string email, string code, CancellationToken cancellationToken)
    {
        var body = $"""
            <p>Your TokenMiner activation code is:</p>
            <p style="font-size:24px;letter-spacing:4px"><strong>{code}</strong></p>
            <p>It expires shortly. If you did not request it, you can ignore this message.</p>
            """;

        return SendAsync(email, "Your TokenMiner activation code", body, cancellationToken);
    }

    private async Task SendAsync(string email, string subject, string htmlBody, CancellationToken cancellationToken)
    {
        var host = smtpOptions.Host;
        if (string.IsNullOrWhiteSpace(host))
        {
            throw new InvalidOperationException(
                "Smtp:Host is not configured; the SMTP sender should not have been selected.");
        }

        var message = new MimeMessage();
        message.From.Add(new MailboxAddress(smtpOptions.FromName, smtpOptions.FromAddress));
        message.To.Add(MailboxAddress.Parse(email));
        message.Subject = subject;
        message.Body = new BodyBuilder { HtmlBody = htmlBody }.ToMessageBody();

        using var client = new SmtpClient();
        await client.ConnectAsync(host, smtpOptions.Port, SecureSocketOptions.StartTlsWhenAvailable, cancellationToken);

        if (!string.IsNullOrWhiteSpace(smtpOptions.UserName))
        {
            await client.AuthenticateAsync(smtpOptions.UserName, smtpOptions.Password ?? string.Empty, cancellationToken);
        }

        await client.SendAsync(message, cancellationToken);
        await client.DisconnectAsync(quit: true, cancellationToken);
    }
}
