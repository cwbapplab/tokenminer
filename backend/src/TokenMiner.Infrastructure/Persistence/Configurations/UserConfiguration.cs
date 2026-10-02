using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Metadata.Builders;
using TokenMiner.Domain.Users;
using TokenMiner.Domain.Users.Enums;

namespace TokenMiner.Infrastructure.Persistence.Configurations;

internal sealed class UserConfiguration : IEntityTypeConfiguration<User>
{
    public void Configure(EntityTypeBuilder<User> builder)
    {
        builder.HasKey(user => user.Id);

        builder.Property(user => user.Email)
            .IsRequired()
            .HasMaxLength(320);

        builder.Property(user => user.NormalizedEmail)
            .IsRequired()
            .HasMaxLength(320);

        builder.HasIndex(user => user.NormalizedEmail).IsUnique();

        builder.Property(user => user.PasswordHash).HasMaxLength(512);

        builder.Property(user => user.GoogleSub).HasMaxLength(64);

        // Postgres treats NULLs as distinct in a unique index, so unlinked accounts coexist fine.
        builder.HasIndex(user => user.GoogleSub).IsUnique();

        builder.Property(user => user.DisplayName).HasMaxLength(128);

        builder.Property(user => user.Status)
            .IsRequired()
            .HasMaxLength(32)
            .HasConversion(
                status => status.ToDbValue(),
                value => UserStatusDbValues.FromDbValue(value));

        builder.Property(user => user.CreatedAt).IsRequired();
        builder.Property(user => user.UpdatedAt).IsRequired();
    }
}
