namespace TokenMiner.Domain.Users.Enums;

public enum UserStatus
{
    PendingActivation = 0,
    Active = 1,
    Suspended = 2,
}

/// <summary>snake_case database representation, e.g. <c>pending_activation</c>.</summary>
public static class UserStatusDbValues
{
    public const string PendingActivation = "pending_activation";
    public const string Active = "active";
    public const string Suspended = "suspended";

    public static string ToDbValue(this UserStatus status) => status switch
    {
        UserStatus.PendingActivation => PendingActivation,
        UserStatus.Active => Active,
        UserStatus.Suspended => Suspended,
        _ => throw new ArgumentOutOfRangeException(nameof(status), status, "Unknown user status."),
    };

    public static UserStatus FromDbValue(string value) => value switch
    {
        PendingActivation => UserStatus.PendingActivation,
        Active => UserStatus.Active,
        Suspended => UserStatus.Suspended,
        _ => throw new ArgumentOutOfRangeException(nameof(value), value, "Unknown user status value."),
    };
}
