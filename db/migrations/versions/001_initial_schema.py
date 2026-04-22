"""initial schema

Revision ID: 001
Revises:
Create Date: 2026-04-22
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import ARRAY, JSONB

revision = "001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "lottery_draws",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("draw_date", sa.Date(), nullable=False),
        sa.Column("prize_1", sa.String(6), nullable=True),
        sa.Column("prize_near_1", ARRAY(sa.String(6)), nullable=True),
        sa.Column("prize_2", ARRAY(sa.String(6)), nullable=True),
        sa.Column("prize_3", ARRAY(sa.String(6)), nullable=True),
        sa.Column("prize_4", ARRAY(sa.String(6)), nullable=True),
        sa.Column("prize_5", ARRAY(sa.String(6)), nullable=True),
        sa.Column("prize_front_3", ARRAY(sa.String(3)), nullable=True),
        sa.Column("prize_back_3", ARRAY(sa.String(3)), nullable=True),
        sa.Column("prize_back_2", sa.String(2), nullable=True),
        sa.Column("source", sa.String(100), nullable=True),
        sa.Column("raw_data", JSONB(), nullable=True),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), server_default=sa.func.now(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("draw_date"),
    )
    op.create_index("ix_lottery_draws_draw_date", "lottery_draws", ["draw_date"])

    op.create_table(
        "lottery_features",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("draw_date", sa.Date(), nullable=False),
        sa.Column("target_type", sa.String(20), nullable=False),
        sa.Column("candidate", sa.String(10), nullable=False),
        sa.Column("features", JSONB(), nullable=False),
        sa.Column("is_winner", sa.Boolean(), nullable=False),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("draw_date", "target_type", "candidate", name="uq_features_draw_target_candidate"),
    )
    op.create_index("ix_lottery_features_draw_date", "lottery_features", ["draw_date"])

    op.create_table(
        "predictions",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("target_draw_date", sa.Date(), nullable=False),
        sa.Column("target_type", sa.String(20), nullable=False),
        sa.Column("model_name", sa.String(100), nullable=False),
        sa.Column("predicted_numbers", JSONB(), nullable=False),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("target_draw_date", "target_type", "model_name", name="uq_prediction_draw_target_model"),
    )
    op.create_index("ix_predictions_target_draw_date", "predictions", ["target_draw_date"])

    op.create_table(
        "model_performance",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("model_name", sa.String(100), nullable=False),
        sa.Column("target_type", sa.String(20), nullable=False),
        sa.Column("eval_draw_date", sa.Date(), nullable=False),
        sa.Column("hit_top1", sa.Boolean(), nullable=True),
        sa.Column("hit_top5", sa.Boolean(), nullable=True),
        sa.Column("hit_top10", sa.Boolean(), nullable=True),
        sa.Column("correct_rank", sa.Integer(), nullable=True),
        sa.Column("metrics", JSONB(), nullable=True),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_model_performance_eval_draw_date", "model_performance", ["eval_draw_date"])


def downgrade() -> None:
    op.drop_table("model_performance")
    op.drop_table("predictions")
    op.drop_table("lottery_features")
    op.drop_table("lottery_draws")
