"""Tests for the EKS Pod Identity Associations lister."""

from __future__ import annotations

from unittest.mock import MagicMock

from botocore.exceptions import ClientError

from aws_bench.resource_management.fastscan.listers.custom_listers import (
    list_eks_pod_identity_associations,
)


class _Paginator:
    def __init__(self, pages):
        self._pages = pages

    def paginate(self, **_kw):
        return iter(self._pages)


def _session_with_clusters_and_associations(
    clusters: list[str], associations_by_cluster: dict[str, list[dict]]
) -> MagicMock:
    """Build a mock session that returns clusters and per-cluster associations."""
    session = MagicMock()
    client = MagicMock()
    session.client.return_value = client

    client.get_paginator.return_value = _Paginator([{"clusters": clusters}])

    def list_pod_identity_associations(**kwargs):
        cluster = kwargs["clusterName"]
        assocs = associations_by_cluster.get(cluster, [])
        return {"associations": assocs}

    client.list_pod_identity_associations.side_effect = list_pod_identity_associations
    return session


class TestListEksPodIdentityAssociations:
    def test_lists_associations_across_clusters(self):
        session = _session_with_clusters_and_associations(
            clusters=["cluster-a", "cluster-b"],
            associations_by_cluster={
                "cluster-a": [{"associationId": "assoc-1"}, {"associationId": "assoc-2"}],
                "cluster-b": [{"associationId": "assoc-3"}],
            },
        )

        result = list_eks_pod_identity_associations(session)

        assert result == [
            "cluster-a|assoc-1",
            "cluster-a|assoc-2",
            "cluster-b|assoc-3",
        ]

    def test_handles_empty_clusters(self):
        session = MagicMock()
        client = MagicMock()
        session.client.return_value = client
        client.get_paginator.return_value = _Paginator([{"clusters": []}])

        result = list_eks_pod_identity_associations(session)

        assert result == []

    def test_skips_vanished_cluster(self):
        session = MagicMock()
        client = MagicMock()
        session.client.return_value = client
        client.get_paginator.return_value = _Paginator([{"clusters": ["alive", "gone"]}])

        def list_pod_identity_associations(**kwargs):
            if kwargs["clusterName"] == "gone":
                raise ClientError(
                    {"Error": {"Code": "ResourceNotFoundException"}},
                    "ListPodIdentityAssociations",
                )
            return {"associations": [{"associationId": "a-1"}]}

        client.list_pod_identity_associations.side_effect = list_pod_identity_associations

        result = list_eks_pod_identity_associations(session)

        assert result == ["alive|a-1"]

    def test_handles_pagination(self):
        session = MagicMock()
        client = MagicMock()
        session.client.return_value = client
        client.get_paginator.return_value = _Paginator([{"clusters": ["cluster-x"]}])

        responses = iter(
            [
                {
                    "associations": [{"associationId": "a-1"}],
                    "nextToken": "token-2",
                },
                {
                    "associations": [{"associationId": "a-2"}],
                },
            ]
        )
        client.list_pod_identity_associations.side_effect = lambda **_kw: next(responses)

        result = list_eks_pod_identity_associations(session)

        assert result == ["cluster-x|a-1", "cluster-x|a-2"]
        assert client.list_pod_identity_associations.call_count == 2
