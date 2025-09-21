import { ColumnLayout, FormField, Input, RadioGroup, SpaceBetween } from "@cloudscape-design/components";

interface ChunkSelectorProps {
  errors: Record<string, string | string[]>;
  data: { 
    chunkSize: number; 
    chunkOverlap: number;
    chunkingStrategy: "recursive" | "file_level";
  };
  submitting: boolean;
  onChange: (
    data: Partial<{ 
      chunkSize: number; 
      chunkOverlap: number;
      chunkingStrategy: "recursive" | "file_level";
    }>
  ) => void;
}

export function ChunkSelectorField(props: ChunkSelectorProps) {
  return (
    <FormField
      label="Text Splitter"
      stretch={true}
      description="Choose how to split your documents into chunks for embedding. File-level chunking treats each file as a single chunk, while recursive splitting creates smaller, overlapping chunks."
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
                label: "File Level",
                value: "file_level",
                description: "Treat each file as a single chunk. Files over 100KB will automatically use recursive splitting.",
              },
            ]}
            value={props.data.chunkingStrategy}
            onChange={({ detail }) => props.onChange({ chunkingStrategy: detail.value as "recursive" | "file_level" })}
          />
        </FormField>

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
