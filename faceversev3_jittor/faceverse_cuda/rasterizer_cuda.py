import jittor as jt

def forward(vertices, textures, normals, render_colors, image_size):
    return jt.code([render_colors.shape], [render_colors.dtype],
    [vertices, textures, normals],
    cuda_header='''
#include <cuda.h>
#include <cuda_runtime.h>

namespace{
    template <typename scalar_t>
    __global__ void forward_cuda_kernel(
            const scalar_t* __restrict__ vertices,
            const scalar_t* __restrict__ textures,
            const scalar_t* __restrict__ normals,
            scalar_t* render_colors,
            int batch_size,
            int image_size,
            int ver_num) {

        const int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= batch_size * ver_num) {
            return;
        }

        const int is = image_size;
        const int nv = ver_num;
        const int bn = i / ver_num;

        const scalar_t *vertex = &vertices[i * 3];
        const scalar_t *texture = &textures[i * 3];
        const scalar_t *normal = &normals[i * 3];

        if (normal[2] < 0) {
            return;
        }

        const int x = (1 - vertex[1]) * is / 2;
        const int y = (vertex[0] + 1) * is / 2;

        for (int i_ = 0; i_ < 3; i_++) {
            for (int j_ = 0; j_ < 3; j_++) {
                const int xi = x - 1 + i_;
                const int yi = y - 1 + j_;
                if (xi >= is or yi >= is or xi < 0 or yi < 0) {
                    continue;
                }
                const int pn = xi * is + yi;

                if (render_colors[(bn * 4 + 3) * (is * is) + pn] > 0.01) {
                    if (vertex[2] > render_colors[(bn * 4 + 3) * (is * is) + pn]) {
                        continue;
                    }
                }

                for (int k = 0; k < 3; k++) {
                    render_colors[(bn * 4 + k) * (is * is) + pn] = texture[k];
                }
                render_colors[(bn * 4 + 3) * (is * is) + pn] = vertex[2];
            }
        }
    }
}
    ''',
    cuda_src=f'''
    @alias(vertices, in0)
    @alias(textures, in1)
    @alias(normals, in2)
    @alias(render_colors, out0)

    cudaMemsetAsync(out0_p, 0, out0->size);

    const auto batch_size = vertices_shape0;
    const auto ver_num = vertices_shape1;

    const int threads = 512;
    const dim3 blocks ((batch_size * ver_num - 1) / threads +1);

    forward_cuda_kernel<float32><<<blocks, threads>>>(
        vertices_p,
        textures_p,
        normals_p,
        render_colors_p,
        batch_size,
        {image_size},
        ver_num);
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) 
        printf("Error in forward: %s", cudaGetErrorString(err));
    ''')

def backward(render_colors, grad_colors, vertices, textures, normals, grad_vertices, grad_textures, image_size):
    return jt.code([grad_vertices.shape, grad_textures.shape], [grad_vertices.dtype, grad_textures.dtype], 
    [render_colors, grad_colors, vertices, textures, normals],
    cuda_header='''
#include <cuda.h>
#include <cuda_runtime.h>

namespace{
    template <typename scalar_t>
    __global__ void backward_cuda_kernel(
            const scalar_t* __restrict__ vertices,
            const scalar_t* __restrict__ textures,
            const scalar_t* __restrict__ normals,
            const scalar_t* __restrict__ render_colors,
            const scalar_t* __restrict__ grad_colors,
            scalar_t* grad_vertices,
            scalar_t* grad_textures,
            int batch_size,
            int image_size,
            int ver_num) {

        const int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= batch_size * ver_num) {
            return;
        }

        const int is = image_size;
        const int nv = ver_num;
        const int bn = i / ver_num;

        const scalar_t *vertex = &vertices[i * 3];
        const scalar_t *texture = &textures[i * 3];
        const scalar_t *normal = &normals[i * 3];

        if (normal[2] < 0) {
            return;
        }

        const int x = (1 - vertex[1]) * is / 2;
        const int y = (vertex[0] + 1) * is / 2;
        scalar_t grad_color[9] = {0};

        for (int i_ = 0; i_ < 3; i_++) {
            for (int j_ = 0; j_ < 3; j_++) {
                const int xi = x - 1 + i_;
                const int yi = y - 1 + j_;
                if (xi >= is or yi >= is or xi < 0 or yi < 0) {
                    continue;
                }
                const int pn = xi * is + yi;

                if (render_colors[(bn * 4 + 3) * (is * is) + pn] < vertex[2] - 0.3) {
                    continue;
                }

                for (int k = 0; k < 3; k++) {
                    grad_color[i_ * 3 + j_] += abs(grad_colors[(bn * 4 + k) * (is * is) + pn]);
                    grad_textures[i * 3 + k] += grad_colors[(bn * 4 + k) * (is * is) + pn];
                }
            }
        }

        scalar_t sum_color = 0;
        for (int i_ = 0; i_ < 9; i_++) {
            sum_color += grad_color[i_];
        }

        for (int i_ = 0; i_ < 3; i_++) {
            for (int j_ = 0; j_ < 3; j_++) {
                grad_vertices[i * 3 + 1] -= (i_ - 1) * abs(grad_color[i_ * 3 + j_]) * sum_color * 100;
                grad_vertices[i * 3 + 0] += (j_ - 1) * abs(grad_color[i_ * 3 + j_]) * sum_color * 100;
            }
        }
    }
}
    ''',
    cuda_src=f'''
    @alias(render_colors, in0)
    @alias(grad_colors, in1)
    @alias(vertices, in2)
    @alias(textures, in3)
    @alias(normals, in4)
    @alias(grad_vertices, out0)
    @alias(grad_textures, out1)

    cudaMemsetAsync(out0_p, 0, out0->size);
    cudaMemsetAsync(out1_p, 0, out1->size);

    const auto batch_size = vertices_shape0;
    const auto ver_num = vertices_shape1;

    const int threads = 512;
    const dim3 blocks ((batch_size * ver_num - 1) / threads +1);

    backward_cuda_kernel<float32><<<blocks, threads>>>(
        vertices_p,
        textures_p,
        normals_p,
        render_colors_p,
        grad_colors_p,
        grad_vertices_p,
        grad_textures_p,
        batch_size,
        {image_size},
        ver_num);
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) 
        printf("Error in backward_soft_rasterize: %s", cudaGetErrorString(err));
    ''')

def render(face_vertices, textures, faces_info, soft_colors, image_size):
    return jt.code([faces_info.shape, soft_colors.shape], [faces_info.dtype, soft_colors.dtype],
    [face_vertices, textures],
    cuda_header='''
    #include <cuda.h>
    #include <cuda_runtime.h>

namespace{

    template <typename scalar_t>
    __device__ __forceinline__ void barycentric_coordinate(scalar_t *w, const scalar_t x, const scalar_t y, const scalar_t *face_info) {
        w[0] = face_info[3 * 0 + 0] * x + face_info[3 * 0 + 1] * y + face_info[3 * 0 + 2];
        w[1] = face_info[3 * 1 + 0] * x + face_info[3 * 1 + 1] * y + face_info[3 * 1 + 2];
        w[2] = face_info[3 * 2 + 0] * x + face_info[3 * 2 + 1] * y + face_info[3 * 2 + 2];
    }


    template <typename scalar_t>
    __device__ __forceinline__ bool check_border(const scalar_t x, const scalar_t y, const scalar_t *face, const scalar_t threshold) {
        return (x > max(max(face[0], face[3]), face[6]) + threshold ||
                x < min(min(face[0], face[3]), face[6]) - threshold ||
                y > max(max(face[1], face[4]), face[7]) + threshold ||
                y < min(min(face[1], face[4]), face[7]) - threshold);
    }


    template <typename scalar_t>
    __device__ __forceinline__ bool check_pixel_inside(const scalar_t *w) {
        return w[0] <= 1 && w[0] >= 0 && w[1] <= 1 && w[1] >= 0 && w[2] <= 1 && w[2] >= 0;
    }

    // triangle preprocessing
    template <typename scalar_t>
    __global__ void forward_soft_rasterize_inv_cuda_kernel(
            const scalar_t* __restrict__ faces,
            scalar_t* faces_info,
            int batch_size,
            int num_faces,
            int image_size) {
        /* batch number, face, number, image size, face[v012][RGB] */
        const int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= batch_size * num_faces) {
            return;
        }
        // const int is = image_size;
        const scalar_t* face = &faces[i * 9];
        scalar_t* face_inv = &faces_info[i * 27];
        scalar_t* face_sym = &faces_info[i * 27+9];
        scalar_t* face_obt = &faces_info[i * 27+18];

        /* return if backside */
        if ((face[7] - face[1]) * (face[3] - face[0]) > (face[4] - face[1]) * (face[6] - face[0]))
            return;
        /* p[num][xy]: x, y is (-1, 1). */
        scalar_t p[3][2];
        for (int num = 0; num < 3; num++) {
            for (int dim = 0; dim < 2; dim++) {
                p[num][dim] = face[3 * num + dim]; // no normalize
            }
        }
        /* compute face_inv */
        scalar_t face_inv_star[9] = {
            p[1][1] - p[2][1], p[2][0] - p[1][0], p[1][0] * p[2][1] - p[2][0] * p[1][1],
            p[2][1] - p[0][1], p[0][0] - p[2][0], p[2][0] * p[0][1] - p[0][0] * p[2][1],
            p[0][1] - p[1][1], p[1][0] - p[0][0], p[0][0] * p[1][1] - p[1][0] * p[0][1]};
        scalar_t face_inv_determinant = (
            p[2][0] * (p[0][1] - p[1][1]) +
            p[0][0] * (p[1][1] - p[2][1]) +
            p[1][0] * (p[2][1] - p[0][1]));
        face_inv_determinant = face_inv_determinant > 0 ? max(face_inv_determinant, 1e-10) : min(face_inv_determinant, -1e-10);
        /* set to global memory */
        for (int k = 0; k < 9; k++) {
            face_inv[k] = face_inv_star[k] / face_inv_determinant;
        }
        /* F * F.T */
        for (int j = 0; j < 3; j++) {
            for (int k = 0; k < 3; k++) {
                face_sym[j * 3 + k] = face[j * 3 + 0] * face[k * 3 + 0] +
                                    face[j * 3 + 1] * face[k * 3 + 1] + 
                                    1;
            }
        }
        /* check if one arc is obt arc */
        for (int k = 0; k < 3; k++) {
            const int k0 = k;
            const int k1 = (k + 1) % 3;
            const int k2 = (k + 2) % 3;
            if ((p[k1][0] - p[k0][0]) * (p[k2][0] - p[k0][0]) + (p[k1][1] - p[k0][1]) * (p[k2][1] - p[k0][1]) < 0) {
                face_obt[k0] = 1;
                break;
            }
        }
    }

    template <typename scalar_t>
    __global__ void forward_soft_rasterize_cuda_kernel(
            const scalar_t* __restrict__ faces,
            const scalar_t* __restrict__ textures,
            const scalar_t* __restrict__ faces_info,
            scalar_t* soft_colors,
            int batch_size,
            int num_faces,
            int image_size) {

        const int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= batch_size * image_size * image_size) {
            return;
        }
        const int is = image_size;
        const int nf = num_faces;
        const int bn = i / (is * is);
        const int pn = i % (is * is);
        const int yi = is - 1 - (pn / is);
        const int xi = pn % is;
        const scalar_t yp = (2. * yi + 1. - is) / is;
        const scalar_t xp = (2. * xi + 1. - is) / is;
        const scalar_t threshold = 0.0001;

        const scalar_t *face = &faces[bn * nf * 9] - 9;
        const scalar_t *texture = &textures[bn * nf * 9] - 9;
        const scalar_t *face_info = &faces_info[bn * nf * 27] - 27;
        scalar_t soft_color[4] = {0., 0., 0., 0.};
        scalar_t depth_min = 10000000;

        for (int fn = 0; fn < nf; fn++) {
            face += 9;
            texture += 9;
            face_info += 27;

            if (check_border(xp, yp, face, threshold)) continue; // triangle too far away from pixel

            scalar_t w[3];
            scalar_t soft_fragment;

            // compute barycentric coordinate w
            barycentric_coordinate(w, xp, yp, face_info);

            // compute probability map based on distance functions
            soft_fragment = check_pixel_inside(w) ? 1. : 0.;
            if (soft_fragment == 0.) continue; // ignore triangle outside of the pixel

            // aggragate for alpha channel
            if (soft_fragment > 0.5) soft_color[3] = 1.;

            const scalar_t zp = 1. / (w[0] / face[2] + w[1] / face[5] + w[2] / face[8]);

            // aggregate for rgb channels
            if (zp < depth_min && check_pixel_inside(w)) {
                depth_min = zp;
                for (int k = 0; k < 3; k++) {
                    soft_color[k] = w[0] * texture[k] + w[1] * texture[3+k] + w[2] * texture[6+k];
                }
            }
        }

        // finalize aggregation
        for (int k = 0; k < 4; k++) {
            soft_colors[(bn * 4 + k) * (is * is) + pn] = soft_color[k];
        }
    }
}
    ''',
    cuda_src=f'''
    @alias(faces, in0)
    @alias(textures, in1)
    @alias(faces_info, out0)
    @alias(soft_colors, out1)

    cudaMemsetAsync(out0_p, 0, out0->size);
    cudaMemsetAsync(out1_p, 0, out1->size);

    const auto batch_size = faces_shape0;
    const auto num_faces = faces_shape1;
    const int threads = 512;
    const dim3 blocks_1 ((batch_size * num_faces - 1) / threads +1);
    const dim3 blocks_2 ((batch_size * {image_size} * {image_size} - 1) / threads +1);
    
    forward_soft_rasterize_inv_cuda_kernel<float32><<<blocks_1, threads>>>(
        faces_p,
        faces_info_p,
        batch_size,
        num_faces,
        {image_size});

    forward_soft_rasterize_cuda_kernel<float32><<<blocks_2, threads>>>(
        faces_p,
        textures_p,
        faces_info_p,
        soft_colors_p,
        batch_size,
        num_faces,
        {image_size});
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) 
        printf("Error in forward_soft_rasterize: %s\\n", cudaGetErrorString(err));
    ''')
