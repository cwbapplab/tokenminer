namespace TokenMiner.Domain.Users;

/// <summary>Stable identifiers for seeded roles, shared by the model seed and role assignment.</summary>
public static class RoleIds
{
    public static readonly Guid User = Guid.Parse("11111111-1111-1111-1111-111111111111");

    public static readonly Guid Admin = Guid.Parse("22222222-2222-2222-2222-222222222222");
}
