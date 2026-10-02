namespace TokenMiner.Infrastructure.Authentication;

/// <summary>Bound from the <c>Smtp</c> configuration section.</summary>
public sealed class SmtpOptions
{
    public const string SectionName = "Smtp";

    public string? Host { get; set; }

    public int Port { get; set; } = 587;

    public string? UserName { get; set; }

    public string? Password { get; set; }

    public string FromAddress { get; set; } = "no-reply@tokenminer.local";

    public string FromName { get; set; } = "TokenMiner";

    /// <summary>When false, email is written to the log instead of being sent.</summary>
    public bool IsConfigured => !string.IsNullOrWhiteSpace(Host);
}
