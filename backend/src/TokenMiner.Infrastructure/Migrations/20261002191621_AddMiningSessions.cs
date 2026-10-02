using System;
using Microsoft.EntityFrameworkCore.Migrations;

#nullable disable

namespace TokenMiner.Infrastructure.Migrations
{
    /// <inheritdoc />
    public partial class AddMiningSessions : Migration
    {
        /// <inheritdoc />
        protected override void Up(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.CreateTable(
                name: "user_hardware_miners",
                columns: table => new
                {
                    id = table.Column<Guid>(type: "uuid", nullable: false),
                    user_hardware_id = table.Column<Guid>(type: "uuid", nullable: false),
                    pool_id = table.Column<Guid>(type: "uuid", nullable: false),
                    coin_id = table.Column<Guid>(type: "uuid", nullable: false),
                    mining_algo_id = table.Column<Guid>(type: "uuid", nullable: false),
                    worker_identifier = table.Column<string>(type: "character varying(128)", maxLength: 128, nullable: false),
                    status = table.Column<string>(type: "character varying(32)", maxLength: 32, nullable: false),
                    started_at = table.Column<DateTimeOffset>(type: "timestamp with time zone", nullable: false),
                    paused_at = table.Column<DateTimeOffset>(type: "timestamp with time zone", nullable: true),
                    stopped_at = table.Column<DateTimeOffset>(type: "timestamp with time zone", nullable: true),
                    last_activity_at = table.Column<DateTimeOffset>(type: "timestamp with time zone", nullable: false),
                    pause_reason = table.Column<string>(type: "character varying(64)", maxLength: 64, nullable: true),
                    stop_reason = table.Column<string>(type: "character varying(64)", maxLength: 64, nullable: true)
                },
                constraints: table =>
                {
                    table.PrimaryKey("pk_user_hardware_miners", x => x.id);
                    table.ForeignKey(
                        name: "fk_user_hardware_miners_coins_coin_id",
                        column: x => x.coin_id,
                        principalTable: "coins",
                        principalColumn: "id",
                        onDelete: ReferentialAction.Restrict);
                    table.ForeignKey(
                        name: "fk_user_hardware_miners_mining_algos_mining_algo_id",
                        column: x => x.mining_algo_id,
                        principalTable: "mining_algos",
                        principalColumn: "id",
                        onDelete: ReferentialAction.Restrict);
                    table.ForeignKey(
                        name: "fk_user_hardware_miners_pools_pool_id",
                        column: x => x.pool_id,
                        principalTable: "pools",
                        principalColumn: "id",
                        onDelete: ReferentialAction.Restrict);
                    table.ForeignKey(
                        name: "fk_user_hardware_miners_user_hardware_user_hardware_id",
                        column: x => x.user_hardware_id,
                        principalTable: "user_hardware",
                        principalColumn: "id",
                        onDelete: ReferentialAction.Cascade);
                });

            migrationBuilder.CreateTable(
                name: "mining_logs",
                columns: table => new
                {
                    id = table.Column<Guid>(type: "uuid", nullable: false),
                    user_id = table.Column<Guid>(type: "uuid", nullable: false),
                    user_hardware_id = table.Column<Guid>(type: "uuid", nullable: false),
                    user_hardware_miner_id = table.Column<Guid>(type: "uuid", nullable: false),
                    pool_id = table.Column<Guid>(type: "uuid", nullable: false),
                    coin_id = table.Column<Guid>(type: "uuid", nullable: false),
                    event_type = table.Column<string>(type: "character varying(64)", maxLength: 64, nullable: false),
                    reason = table.Column<string>(type: "character varying(256)", maxLength: 256, nullable: true),
                    metadata = table.Column<string>(type: "jsonb", nullable: true),
                    created_at = table.Column<DateTimeOffset>(type: "timestamp with time zone", nullable: false)
                },
                constraints: table =>
                {
                    table.PrimaryKey("pk_mining_logs", x => x.id);
                    table.ForeignKey(
                        name: "fk_mining_logs_coins_coin_id",
                        column: x => x.coin_id,
                        principalTable: "coins",
                        principalColumn: "id",
                        onDelete: ReferentialAction.Restrict);
                    table.ForeignKey(
                        name: "fk_mining_logs_pools_pool_id",
                        column: x => x.pool_id,
                        principalTable: "pools",
                        principalColumn: "id",
                        onDelete: ReferentialAction.Restrict);
                    table.ForeignKey(
                        name: "fk_mining_logs_user_hardware_miners_user_hardware_miner_id",
                        column: x => x.user_hardware_miner_id,
                        principalTable: "user_hardware_miners",
                        principalColumn: "id",
                        onDelete: ReferentialAction.Cascade);
                    table.ForeignKey(
                        name: "fk_mining_logs_user_hardware_user_hardware_id",
                        column: x => x.user_hardware_id,
                        principalTable: "user_hardware",
                        principalColumn: "id",
                        onDelete: ReferentialAction.Cascade);
                    table.ForeignKey(
                        name: "fk_mining_logs_users_user_id",
                        column: x => x.user_id,
                        principalTable: "users",
                        principalColumn: "id",
                        onDelete: ReferentialAction.Cascade);
                });

            migrationBuilder.CreateIndex(
                name: "ix_mining_logs_coin_id",
                table: "mining_logs",
                column: "coin_id");

            migrationBuilder.CreateIndex(
                name: "ix_mining_logs_pool_id",
                table: "mining_logs",
                column: "pool_id");

            migrationBuilder.CreateIndex(
                name: "ix_mining_logs_user_hardware_id",
                table: "mining_logs",
                column: "user_hardware_id");

            migrationBuilder.CreateIndex(
                name: "ix_mining_logs_user_hardware_miner_id",
                table: "mining_logs",
                column: "user_hardware_miner_id");

            migrationBuilder.CreateIndex(
                name: "ix_mining_logs_user_id_created_at",
                table: "mining_logs",
                columns: new[] { "user_id", "created_at" });

            migrationBuilder.CreateIndex(
                name: "ix_user_hardware_miners_coin_id",
                table: "user_hardware_miners",
                column: "coin_id");

            migrationBuilder.CreateIndex(
                name: "ix_user_hardware_miners_mining_algo_id",
                table: "user_hardware_miners",
                column: "mining_algo_id");

            migrationBuilder.CreateIndex(
                name: "ix_user_hardware_miners_pool_id",
                table: "user_hardware_miners",
                column: "pool_id");

            migrationBuilder.CreateIndex(
                name: "ix_user_hardware_miners_status_last_activity_at",
                table: "user_hardware_miners",
                columns: new[] { "status", "last_activity_at" });

            migrationBuilder.CreateIndex(
                name: "ix_user_hardware_miners_user_hardware_id",
                table: "user_hardware_miners",
                column: "user_hardware_id",
                unique: true,
                filter: "status IN ('running', 'paused')");
        }

        /// <inheritdoc />
        protected override void Down(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.DropTable(
                name: "mining_logs");

            migrationBuilder.DropTable(
                name: "user_hardware_miners");
        }
    }
}
