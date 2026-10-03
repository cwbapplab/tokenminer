using Microsoft.EntityFrameworkCore.Migrations;

#nullable disable

namespace TokenMiner.Infrastructure.Migrations
{
    /// <inheritdoc />
    public partial class EnforceUniqueIdentifiers : Migration
    {
        /// <inheritdoc />
        protected override void Up(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.DropIndex(
                name: "ix_user_mining_shares_pool_id_share_identifier",
                table: "user_mining_shares");

            migrationBuilder.DropIndex(
                name: "ix_pool_payouts_pool_id_transaction_hash",
                table: "pool_payouts");

            migrationBuilder.CreateIndex(
                name: "ix_user_mining_shares_pool_id",
                table: "user_mining_shares",
                column: "pool_id");

            migrationBuilder.CreateIndex(
                name: "ix_user_mining_shares_share_identifier",
                table: "user_mining_shares",
                column: "share_identifier",
                unique: true);

            migrationBuilder.CreateIndex(
                name: "ix_provider_deposits_transaction_hash",
                table: "provider_deposits",
                column: "transaction_hash",
                unique: true);

            migrationBuilder.CreateIndex(
                name: "ix_pool_payouts_pool_id",
                table: "pool_payouts",
                column: "pool_id");

            migrationBuilder.CreateIndex(
                name: "ix_pool_payouts_transaction_hash",
                table: "pool_payouts",
                column: "transaction_hash",
                unique: true);
        }

        /// <inheritdoc />
        protected override void Down(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.DropIndex(
                name: "ix_user_mining_shares_pool_id",
                table: "user_mining_shares");

            migrationBuilder.DropIndex(
                name: "ix_user_mining_shares_share_identifier",
                table: "user_mining_shares");

            migrationBuilder.DropIndex(
                name: "ix_provider_deposits_transaction_hash",
                table: "provider_deposits");

            migrationBuilder.DropIndex(
                name: "ix_pool_payouts_pool_id",
                table: "pool_payouts");

            migrationBuilder.DropIndex(
                name: "ix_pool_payouts_transaction_hash",
                table: "pool_payouts");

            migrationBuilder.CreateIndex(
                name: "ix_user_mining_shares_pool_id_share_identifier",
                table: "user_mining_shares",
                columns: new[] { "pool_id", "share_identifier" },
                unique: true);

            migrationBuilder.CreateIndex(
                name: "ix_pool_payouts_pool_id_transaction_hash",
                table: "pool_payouts",
                columns: new[] { "pool_id", "transaction_hash" },
                unique: true);
        }
    }
}
