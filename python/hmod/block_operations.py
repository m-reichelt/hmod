import numpy as np
from scipy.sparse.linalg import LinearOperator


class BlockLinearOperator(LinearOperator):
    """
    Block operator built from a rectangular list of lists of LinearOperator or None.

    Parameters
    ----------
    blocks : list[list[LinearOperator | None]]
        Rectangular block layout. `None` means a zero block.
    dtype : dtype, optional
        If omitted, inferred from the block dtypes.

    Notes
    -----
    - Every block row must contain at least one non-None block, otherwise its row size
      is ambiguous.
    - Every block column must contain at least one non-None block, otherwise its column
      size is ambiguous.
    - Supports matvec, matmat, rmatvec, and rmatmat.
    """

    def __init__(self, blocks, dtype=None):
        self.blocks = self._validate_block_grid(blocks)
        self.n_block_rows = len(self.blocks)
        self.n_block_cols = len(self.blocks[0])

        self.row_sizes, self.col_sizes = self._infer_and_check_block_sizes(self.blocks)

        shape = (sum(self.row_sizes), sum(self.col_sizes))
        dtype = self._infer_dtype(dtype, self.blocks)

        self._row_offsets = np.cumsum([0] + self.row_sizes)
        self._col_offsets = np.cumsum([0] + self.col_sizes)

        super().__init__(dtype=dtype, shape=shape)

    @staticmethod
    def _validate_block_grid(blocks):
        if not isinstance(blocks, (list, tuple)) or len(blocks) == 0:
            raise ValueError("blocks must be a non-empty list of lists")

        rows = [list(row) for row in blocks]
        ncols = len(rows[0])
        if ncols == 0:
            raise ValueError("blocks must have at least one column")

        for i, row in enumerate(rows):
            if len(row) != ncols:
                raise ValueError(
                    f"blocks must form a rectangular grid, but row 0 has length {ncols} "
                    f"and row {i} has length {len(row)}"
                )
            for j, op in enumerate(row):
                if op is not None and not isinstance(op, LinearOperator):
                    raise TypeError(
                        f"blocks[{i}][{j}] must be a LinearOperator or None, got {type(op)}"
                    )
        return rows

    @staticmethod
    def _infer_dtype(dtype, blocks):
        if dtype is not None:
            return np.dtype(dtype)

        dtypes = []
        for row in blocks:
            for op in row:
                if op is not None and getattr(op, "dtype", None) is not None:
                    dtypes.append(np.dtype(op.dtype))

        return np.result_type(*dtypes) if dtypes else np.dtype(float)

    @staticmethod
    def _infer_and_check_block_sizes(blocks):
        n_block_rows = len(blocks)
        n_block_cols = len(blocks[0])

        row_sizes = [None] * n_block_rows
        col_sizes = [None] * n_block_cols

        # Infer row sizes and check consistency of block row heights
        for i in range(n_block_rows):
            for j in range(n_block_cols):
                op = blocks[i][j]
                if op is None:
                    continue
                m, n = op.shape

                if row_sizes[i] is None:
                    row_sizes[i] = m
                elif row_sizes[i] != m:
                    raise ValueError(
                        f"Inconsistent row block sizes in block row {i}: "
                        f"expected {row_sizes[i]}, got {m} at block ({i}, {j})"
                    )

                if col_sizes[j] is None:
                    col_sizes[j] = n
                elif col_sizes[j] != n:
                    raise ValueError(
                        f"Inconsistent column block sizes in block column {j}: "
                        f"expected {col_sizes[j]}, got {n} at block ({i}, {j})"
                    )

        for i, size in enumerate(row_sizes):
            if size is None:
                raise ValueError(
                    f"Cannot infer size of block row {i}: entire row is None"
                )

        for j, size in enumerate(col_sizes):
            if size is None:
                raise ValueError(
                    f"Cannot infer size of block column {j}: entire column is None"
                )

        return row_sizes, col_sizes

    def _split_input_columns(self, X):
        """
        Split X of shape (N, k) into block-column pieces.
        """
        pieces = []
        for j in range(self.n_block_cols):
            a = self._col_offsets[j]
            b = self._col_offsets[j + 1]
            pieces.append(X[a:b, :])
        return pieces

    def _split_input_rows(self, X):
        """
        Split X of shape (M, k) into block-row pieces.
        """
        pieces = []
        for i in range(self.n_block_rows):
            a = self._row_offsets[i]
            b = self._row_offsets[i + 1]
            pieces.append(X[a:b, :])
        return pieces

    def _matmat(self, X):
        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError(f"_matmat expects a 2D array, got shape {X.shape}")
        if X.shape[0] != self.shape[1]:
            raise ValueError(
                f"dimension mismatch in _matmat: operator shape {self.shape}, "
                f"got X with shape {X.shape}"
            )

        k = X.shape[1]
        x_blocks = self._split_input_columns(X)
        Y = np.zeros((self.shape[0], k), dtype=self.dtype)

        for i in range(self.n_block_rows):
            ya = self._row_offsets[i]
            yb = self._row_offsets[i + 1]
            Yi = Y[ya:yb, :]

            for j in range(self.n_block_cols):
                op = self.blocks[i][j]
                if op is None:
                    continue
                Yi += op.matmat(x_blocks[j])

        return Y

    def _matvec(self, x):
        x = np.asarray(x)
        if x.ndim != 1:
            x = x.reshape(-1)
        return self._matmat(x[:, None])[:, 0]

    def _rmatmat(self, X):
        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError(f"_rmatmat expects a 2D array, got shape {X.shape}")
        if X.shape[0] != self.shape[0]:
            raise ValueError(
                f"dimension mismatch in _rmatmat: operator shape {self.shape}, "
                f"got X with shape {X.shape}"
            )

        k = X.shape[1]
        x_blocks = self._split_input_rows(X)
        Y = np.zeros((self.shape[1], k), dtype=self.dtype)

        for j in range(self.n_block_cols):
            ya = self._col_offsets[j]
            yb = self._col_offsets[j + 1]
            Yj = Y[ya:yb, :]

            for i in range(self.n_block_rows):
                op = self.blocks[i][j]
                if op is None:
                    continue

                # Prefer rmatmat if available, otherwise use adjoint matmat
                if hasattr(op, "rmatmat"):
                    Yj += op.rmatmat(x_blocks[i])
                else:
                    Yj += op.H.matmat(x_blocks[i])

        return Y

    def _rmatvec(self, x):
        x = np.asarray(x)
        if x.ndim != 1:
            x = x.reshape(-1)
        return self._rmatmat(x[:, None])[:, 0]

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"shape={self.shape}, dtype={self.dtype}, "
            f"block_shape=({self.n_block_rows}, {self.n_block_cols}))"
        )