package media

import "fmt"

const (
	// AlignedChunkSize is the strict MTProto upload.getFile part size (512 KB).
	AlignedChunkSize int64 = 524288
	// MinPartUnit enforces Telegram's 1 KB granularity requirement.
	MinPartUnit = 1024
)

// AlignedChunk maps one 512 KB MTProto upload.getFile request to the requested HTTP byte slice.
type AlignedChunk struct {
	Offset     int64 // Always an exact multiple of 524288
	Limit      int   // Always 524288 (satisfies part_size % 1024 == 0 and 524288 % part_size == 0)
	SliceStart int   // Inclusive start offset inside the returned 512 KB chunk
	SliceEnd   int   // Exclusive upper bound inside the returned chunk
}

// ValidateUploadPartParams verifies Telegram's strict upload.getFile alignment constraints
// from SPEC Section 7.5.4:
//   - partSize > 0
//   - partSize % 1024 == 0
//   - 524288 % partSize == 0
//   - offset >= 0 and offset % int64(partSize) == 0
func ValidateUploadPartParams(offset int64, partSize int) error {
	if partSize <= 0 {
		return fmt.Errorf("invalid upload.getFile part_size %d: must be > 0", partSize)
	}
	if partSize%MinPartUnit != 0 {
		return fmt.Errorf("invalid upload.getFile part_size %d: must satisfy part_size %% 1024 == 0", partSize)
	}
	if int(AlignedChunkSize)%partSize != 0 {
		return fmt.Errorf("invalid upload.getFile part_size %d: must satisfy 524288 %% part_size == 0", partSize)
	}
	if offset < 0 {
		return fmt.Errorf("invalid upload.getFile offset %d: must be >= 0", offset)
	}
	if offset%int64(partSize) != 0 {
		return fmt.Errorf("invalid upload.getFile offset %d: must be aligned to part_size %d", offset, partSize)
	}
	return nil
}

// AlignChunkRange converts an arbitrary inclusive HTTP byte range [reqStart, reqEnd] into
// aligned 512 KB MTProto chunks with exact sub-slice bounds.
func AlignChunkRange(reqStart, reqEnd, totalSize int64) ([]AlignedChunk, error) {
	if reqStart < 0 || reqEnd < reqStart {
		return nil, fmt.Errorf("invalid byte range [%d, %d]", reqStart, reqEnd)
	}
	if totalSize > 0 {
		if reqStart >= totalSize {
			return nil, fmt.Errorf("range start %d exceeds file size %d", reqStart, totalSize)
		}
		if reqEnd >= totalSize {
			reqEnd = totalSize - 1
		}
	}

	firstChunk := reqStart / AlignedChunkSize
	lastChunk := reqEnd / AlignedChunkSize

	chunks := make([]AlignedChunk, 0, lastChunk-firstChunk+1)
	for idx := firstChunk; idx <= lastChunk; idx++ {
		chunkOffset := idx * AlignedChunkSize
		chunkLimit := int(AlignedChunkSize)

		if err := ValidateUploadPartParams(chunkOffset, chunkLimit); err != nil {
			return nil, err
		}

		chunkEndByte := chunkOffset + AlignedChunkSize - 1
		sliceStart := 0
		if reqStart > chunkOffset {
			sliceStart = int(reqStart - chunkOffset)
		}
		sliceEnd := chunkLimit
		if reqEnd < chunkEndByte {
			sliceEnd = int(reqEnd-chunkOffset) + 1
		}

		chunks = append(chunks, AlignedChunk{
			Offset:     chunkOffset,
			Limit:      chunkLimit,
			SliceStart: sliceStart,
			SliceEnd:   sliceEnd,
		})
	}
	return chunks, nil
}
