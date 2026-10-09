using System;
using System.Collections.Generic;
using MathNet.Numerics.LinearAlgebra;

public static class NastyNoiseInjector
{
    public static void InjectNastyNoise(List<DataRow> data, double a, double b, float labelNoiseFraction, float featureNoiseFraction)
    {
        Console.WriteLine("Injecting nasty noise...");

        // Step 1: Label Noise
        InjectLabelNoise(data, labelNoiseFraction);

        // Step 2: Random Feature Distortion
        InjectRandomFeatureNoise(data, featureNoiseFraction);

        // Step 3: Non-linear Transformation (Hyperbolic + Inverse Tangent)
        InjectHyperbolicAndInverseTangentNoise(data, a, b);

        // Step 4: Cholesky Decomposition
        ApplyCholeskyDecomposition(data);

        Console.WriteLine("Nasty noise injection complete!");
    }

    private static void InjectLabelNoise(List<DataRow> data, float noiseFraction)
    {
        var random = new Random();
        int noiseCount = (int)(data.Count * noiseFraction);

        for (int i = 0; i < noiseCount; i++)
        {
            int index = random.Next(data.Count);
            data[index].Label = data[index].Label == "ClassA" ? "ClassB" : "ClassA";
        }

        Console.WriteLine($"Injected label noise into {noiseCount} rows.");
    }

    private static void InjectRandomFeatureNoise(List<DataRow> data, float noiseFraction)
    {
        var random = new Random();
        int noiseCount = (int)(data.Count * noiseFraction);

        for (int i = 0; i < noiseCount; i++)
        {
            int index = random.Next(data.Count);
            data[index].Feature1 += random.NextDouble() * 5.0 - 2.5; // Add random noise
            data[index].Feature2 *= random.Next(1, 4); // Multiply feature randomly
        }

        Console.WriteLine($"Injected random feature noise into {noiseCount} rows.");
    }

    private static void InjectHyperbolicAndInverseTangentNoise(List<DataRow> data, double a, double b)
    {
        var random = new Random();
        foreach (var row in data)
        {
            // Hyperbolic transformation
            row.Feature1 = (float)Math.Sqrt((Math.Pow(row.Feature1, 2) / Math.Pow(a, 2)) + 1);
            row.Feature2 = (float)Math.Sqrt((Math.Pow(row.Feature2, 2) / Math.Pow(b, 2)) - 1);

            // Inverse tangent transformation
            row.Feature1 = (float)Math.Atan(row.Feature1);
            row.Feature2 = (float)Math.Atan(row.Feature2);

            // Add small random perturbations
            row.Feature1 += (float)(random.NextDouble() * 0.1 - 0.05);
            row.Feature2 += (float)(random.NextDouble() * 0.1 - 0.05);
        }

        Console.WriteLine("Applied hyperbolic and inverse tangent transformations.");
    }

    private static void ApplyCholeskyDecomposition(List<DataRow> data)
    {
        // Extract features into arrays
        var feature1Values = data.ConvertAll(row => (double)row.Feature1).ToArray();
        var feature2Values = data.ConvertAll(row => (double)row.Feature2).ToArray();

        // Construct covariance matrix
        var covarianceMatrix = Matrix<double>.Build.DenseOfArray(new double[,]
        {
            { Variance(feature1Values), Covariance(feature1Values, feature2Values) },
            { Covariance(feature1Values, feature2Values), Variance(feature2Values) }
        });

        // Perform Cholesky decomposition
        var lowerTriangularMatrix = covarianceMatrix.Cholesky().Factor;

        // Transform each feature vector
        foreach (var row in data)
        {
            var featureVector = Vector<double>.Build.DenseOfArray(new double[] { row.Feature1, row.Feature2 });
            var transformedVector = lowerTriangularMatrix * featureVector;

            row.Feature1 = (float)transformedVector[0];
            row.Feature2 = (float)transformedVector[1];
        }

        Console.WriteLine("Applied Cholesky decomposition.");
    }

    // Helper: Calculate variance
    private static double Variance(double[] values)
    {
        double mean = values.Average();
        return values.Select(v => Math.Pow(v - mean, 2)).Average();
    }

    // Helper: Calculate covariance
    private static double Covariance(double[] x, double[] y)
    {
        double meanX = x.Average();
        double meanY = y.Average();
        return x.Zip(y, (xi, yi) => (xi - meanX) * (yi - meanY)).Average();
    }
}
