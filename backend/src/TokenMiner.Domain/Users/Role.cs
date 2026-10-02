namespace TokenMiner.Domain.Users;

public sealed class Role
{
    public const string User = "user";
    public const string Admin = "admin";

    private Role()
    {
    }

    public Role(Guid id, string name)
    {
        Id = id;
        Name = name;
    }

    public Guid Id { get; private set; }

    public string Name { get; private set; } = null!;
}
