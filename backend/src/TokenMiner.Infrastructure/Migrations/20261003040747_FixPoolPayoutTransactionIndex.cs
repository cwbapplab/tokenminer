using Microsoft.EntityFrameworkCore.Migrations;

#nullable disable

namespace TokenMiner.Infrastructure.Migrations
{
    /// <inheritdoc />
    public partial class FixPoolPayoutTransactionIndex : Migration
    {
        /// <inheritdoc />
        protected override void Up(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.DropIndex(
                name: "ix_pool_payouts_pool_id",
                table: "pool_payouts");

            migrationBuilder.DropIndex(
                name: "ix_pool_payouts_transaction_hash",
                table: "pool_payouts");

            migrationBuilder.CreateIndex(
                name: "ix_pool_payouts_pool_id_transaction_hash",
                table: "pool_payouts",
                columns: new[] { "pool_id", "transaction_hash" },
                unique: true);
        }

        /// <inheritdoc />
        protected override void Down(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.DropIndex(
                name: "ix_pool_payouts_pool_id_transaction_hash",
                table: "pool_payouts");

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
    }
}
