"""Tests for the EKS Nodegroups lister."""

from __future__ import annotations

from unittest.mock import MagicMock

from botocore.exceptions import ClientError

from aws_bench.resource_management.fastscan.listers.custom_listers import (
    list_eks_nodegroups,
)


class _Paginator:
    def __init__(self, pages):
        self._pages = pages

    def paginate(self, **_kw):
        return iter(self._pages)


class TestListEksNodegroups:
    def test_lists_nodegroups_across_clusters(self):
        session = MagicMock()
        client = MagicMock()
        session.client.return_value = client

        cluster_paginator = _Paginator([{"clusters": ["cluster-a", "cluster-b"]}])
        nodegroup_paginator_a = _Paginator([{"nodegroups": ["ng-1", "ng-2"]}])
        nodegroup_paginator_b = _Paginator([{"nodegroups": ["ng-3"]}])

        paginators = iter([cluster_paginator, nodegroup_paginator_a, nodegroup_paginator_b])
        client.get_paginator.side_effect = lambda op: next(paginators)

        result = list_eks_nodegroups(session)

        assert result == [
            "cluster-a|ng-1",
            "cluster-a|ng-2",
            "cluster-b|ng-3",
        ]

    def test_handles_empty_clusters(self):
        session = MagicMock()
        client = MagicMock()
        session.client.return_value = client
        client.get_paginator.return_value = _Paginator([{"clusters": []}])

        result = list_eks_nodegroups(session)

        assert result == []

    def test_skips_vanished_cluster(self):
        session = MagicMock()
        client = MagicMock()
        session.client.return_value = client

        class _NodegroupPaginator:
            def paginate(self, **kwargs):
                if kwargs.get("clusterName") == "gone":
                    raise ClientError(
                        {"Error": {"Code": "ResourceNotFoundException"}},
                        "ListNodegroups",
                    )
                return iter([{"nodegroups": ["ng-ok"]}])

        def get_paginator(op):
            if op == "list_clusters":
                return _Paginator([{"clusters": ["alive", "gone"]}])
            return _NodegroupPaginator()

        client.get_paginator.side_effect = get_paginator

        result = list_eks_nodegroups(session)

        assert result == ["alive|ng-ok"]

    def test_handles_pagination(self):
        session = MagicMock()
        client = MagicMock()
        session.client.return_value = client

        cluster_paginator = _Paginator([{"clusters": ["cluster-x"]}])
        nodegroup_paginator = _Paginator([{"nodegroups": ["ng-1"]}, {"nodegroups": ["ng-2"]}])

        def get_paginator(op):
            if op == "list_clusters":
                return cluster_paginator
            return nodegroup_paginator

        client.get_paginator.side_effect = get_paginator

        result = list_eks_nodegroups(session)

        assert result == ["cluster-x|ng-1", "cluster-x|ng-2"]
