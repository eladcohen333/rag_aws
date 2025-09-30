import { Box, ColumnLayout, FormField, Input, RadioGroup, SpaceBetween } from "@cloudscape-design/components";

interface ChunkSelectorProps {
  errors: Record<string, string | string[]>;
  data: { 
    chunkSize: number; 
    chunkOverlap: number;
    chunkingStrategy: "recursive" | "file_level" | "semantic";
  };
  submitting: boolean;
  onChange: (
    data: Partial<{ 
      chunkSize: number; 
      chunkOverlap: number;
      chunkingStrategy: "recursive" | "file_level" | "semantic";
    }>
  ) => void;
}

export function ChunkSelectorField(props: ChunkSelectorProps) {
  return (
    <FormField
      label="Text Splitter"
      stretch={true}
      description="Choose how to split your documents into chunks for embedding. Semantic chunking uses embeddings to find natural boundaries, recursive splitting creates smaller overlapping chunks, and file-level treats each file as a single chunk."
    >
      <SpaceBetween size="l">
        <FormField label="Chunking Strategy" errorText={props.errors.chunkingStrategy}>
          <RadioGroup
            items={[
              {
                label: "Recursive (Recommended for most cases)",
                value: "recursive",
                description: "Split text into smaller chunks with overlap. Better for precise retrieval.",
              },
              {
                label: "Semantic (Preview)",
                value: "semantic",
                description: "Use embeddings-aware semantic boundaries when splitting content.",
              },
              {
                label: "File Level",
                value: "file_level",
                description: "Treat each file as a single chunk. Files over 100KB will automatically use recursive splitting.",
              },
            ]}
            value={props.data.chunkingStrategy}
            onChange={({ detail }) => props.onChange({ chunkingStrategy: detail.value as "recursive" | "file_level" | "semantic" })}
          />
        </FormField>

        {props.data.chunkingStrategy === "semantic" && (
          <Box color="text-body-secondary">
            Semantic chunking automatically determines chunk sizes; the values below are ignored when this option is selected.
          </Box>
        )}

        {props.data.chunkingStrategy === "recursive" && (
          <ColumnLayout columns={2}>
            <FormField label="Chunk Size" errorText={props.errors.chunkSize}>
              <Input
                type="number"
                disabled={props.submitting}
                value={props.data.chunkSize.toString()}
                onChange={({ detail: { value } }) =>
                  props.onChange({ chunkSize: parseInt(value) })
                }
              />
            </FormField>
            <FormField label="Chunk Overlap" errorText={props.errors.chunkOverlap}>
              <Input
                type="number"
                disabled={props.submitting}
                value={props.data.chunkOverlap.toString()}
                onChange={({ detail: { value } }) =>
                  props.onChange({ chunkOverlap: parseInt(value) })
                }
              />
            </FormField>
          </ColumnLayout>
        )}
      </SpaceBetween>
    </FormField>
  );
}
