import { useEffect, useCallback, useMemo } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  ReactFlow,
  Node,
  Edge,
  Controls,
  Background,
  useNodesState,
  useEdgesState,
  MarkerType,
  Position,
} from '@xyflow/react';
import '@xyflow/react/dist/style.css';

import { useCurriculumStore, ERA_INFO } from '@/stores/curriculumStore';
import { useProgressStore } from '@/stores/progressStore';
import { useAuthStore } from '@/stores/authStore';

// Custom node component
function TopicNode({ data }: { data: { label: string; era: string; progress?: number; slug: string } }) {
  const eraInfo = ERA_INFO[data.era] || { color: '#6B7280' };

  return (
    <div
      className="px-4 py-3 rounded-lg shadow-md border-2 bg-white min-w-[150px] max-w-[200px] cursor-pointer hover:shadow-lg transition-shadow"
      style={{ borderColor: eraInfo.color }}
    >
      <div
        className="w-full h-1 rounded-full mb-2"
        style={{ backgroundColor: eraInfo.color }}
      />
      <div className="text-sm font-medium text-gray-900 text-center leading-tight">
        {data.label}
      </div>
      {data.progress !== undefined && data.progress > 0 && (
        <div className="mt-2">
          <div className="w-full h-1 bg-gray-200 rounded-full overflow-hidden">
            <div
              className="h-full bg-green-500 rounded-full transition-all"
              style={{ width: `${data.progress}%` }}
            />
          </div>
          <div className="text-xs text-center text-gray-500 mt-1">
            {data.progress}%
          </div>
        </div>
      )}
    </div>
  );
}

const nodeTypes = {
  topic: TopicNode,
};

export function MindMap() {
  const navigate = useNavigate();
  const { topics, connections, fetchTopics, fetchConnections } = useCurriculumStore();
  const { overview, fetchOverview } = useProgressStore();
  const { isAuthenticated } = useAuthStore();

  const [nodes, setNodes, onNodesChange] = useNodesState<Node>([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState<Edge>([]);

  useEffect(() => {
    fetchTopics();
    fetchConnections();
    if (isAuthenticated) {
      fetchOverview();
    }
  }, [fetchTopics, fetchConnections, fetchOverview, isAuthenticated]);

  // Create progress lookup
  const progressLookup = useMemo(() => {
    return overview.reduce((acc, item) => {
      acc[item.slug] = item.percentComplete;
      return acc;
    }, {} as Record<string, number>);
  }, [overview]);

  // Generate node positions based on era and order
  useEffect(() => {
    if (topics.length === 0) return;

    // Group by era
    const topicsByEra: Record<string, typeof topics> = {};
    topics.forEach((topic) => {
      const era = topic.era || 'unknown';
      if (!topicsByEra[era]) topicsByEra[era] = [];
      topicsByEra[era].push(topic);
    });

    // Position nodes
    const newNodes: Node[] = [];
    const eraOrder = ['foundations', 'ai-winter', 'ml-renaissance', 'deep-learning', 'modern-ai'];
    const nodeSpacingX = 280;
    const nodeSpacingY = 120;

    eraOrder.forEach((era, eraIndex) => {
      const eraTopics = topicsByEra[era] || [];
      const eraX = eraIndex * nodeSpacingX;

      eraTopics.forEach((topic, topicIndex) => {
        const topicY = topicIndex * nodeSpacingY;

        newNodes.push({
          id: topic.slug,
          type: 'topic',
          position: { x: eraX, y: topicY },
          data: {
            label: topic.title,
            era: topic.era || 'unknown',
            progress: isAuthenticated ? progressLookup[topic.slug] : undefined,
            slug: topic.slug,
          },
          sourcePosition: Position.Right,
          targetPosition: Position.Left,
        });
      });
    });

    setNodes(newNodes);
  }, [topics, progressLookup, isAuthenticated, setNodes]);

  // Generate edges from connections
  useEffect(() => {
    if (connections.length === 0) return;

    const newEdges: Edge[] = connections.map((conn) => ({
      id: `${conn.fromSlug}-${conn.toSlug}`,
      source: conn.fromSlug,
      target: conn.toSlug,
      label: conn.label || undefined,
      type: 'smoothstep',
      animated: conn.connectionType === 'leads_to',
      style: { stroke: '#94a3b8', strokeWidth: 2 },
      labelStyle: { fontSize: 10, fill: '#64748b' },
      markerEnd: {
        type: MarkerType.ArrowClosed,
        color: '#94a3b8',
      },
    }));

    setEdges(newEdges);
  }, [connections, setEdges]);

  const onNodeClick = useCallback(
    (_: React.MouseEvent, node: Node) => {
      navigate(`/learn/${node.id}`);
    },
    [navigate]
  );

  if (topics.length === 0) {
    return (
      <div className="flex items-center justify-center h-[600px]">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary-600" />
      </div>
    );
  }

  return (
    <div className="h-[600px] w-full border border-gray-200 rounded-lg overflow-hidden bg-gray-50">
      <ReactFlow
        nodes={nodes}
        edges={edges}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        onNodeClick={onNodeClick}
        nodeTypes={nodeTypes}
        fitView
        fitViewOptions={{ padding: 0.2 }}
        minZoom={0.3}
        maxZoom={1.5}
        defaultEdgeOptions={{
          type: 'smoothstep',
        }}
      >
        <Controls />
        <Background color="#e2e8f0" gap={20} />
      </ReactFlow>

      {/* Legend */}
      <div className="absolute bottom-4 left-4 bg-white p-3 rounded-lg shadow-md border border-gray-200">
        <div className="text-xs font-medium text-gray-700 mb-2">Eras</div>
        <div className="space-y-1">
          {Object.entries(ERA_INFO).map(([key, info]) => (
            <div key={key} className="flex items-center gap-2 text-xs">
              <div
                className="w-3 h-3 rounded-full"
                style={{ backgroundColor: info.color }}
              />
              <span className="text-gray-600">{info.name.split('(')[0].trim()}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
