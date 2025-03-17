import os
import pandas as pd
import seaborn as sns
from PIL import Image
from constants import *
import matplotlib.pyplot as plt

Image.MAX_IMAGE_PIXELS = None

def plotHourlyData(df, df_name, time_column, value_column):
    try:
        regionName = str(str(df_name).split('_')[0])
        fileName = str(str(df_name).split('_')[-1])
        targetFile = os.path.join(plotsFolder+'/'+regionName, fileName)
        # Extract the start time from the 'MTU (CET/CEST)' column
        df['Timestamp'] = pd.to_datetime(df[time_column].str.split(' - ').str[0], format='%d.%m.%Y %H:%M')

        # Sort by time (optional but ensures correct order)
        df = df.sort_values(by='Timestamp')

        # Plot the data
        plt.figure(figsize=(12, 6))
        plt.plot(df['Timestamp'], df[value_column], marker='o', linestyle='-', color='b', label=value_column)

        # Formatting the plot
        plt.xlabel("Time (Hourly)")
        plt.ylabel(value_column)
        plt.title(f"{fileName.replace('.png','')}")
        plt.xticks(rotation=45)  # Rotate labels for better readability
        plt.legend()
        plt.grid(True)

        # Save the plot as a high-resolution PNG file
        plt.savefig(targetFile, dpi=1200, bbox_inches='tight')

        # Show the plot
        # plt.show()
        plt.close()
    
    except Exception as e:
        print(f"Plot Hourly Rates Error: {e}")

def readCsv(filePath: str):
    try:
        # Read CSV into a DataFrame
        df = pd.read_csv(filePath)
        
        # Print Data
        print(f"\nTitles : {df.columns.tolist()}")
        print(df.head())  # Print first few rows
        print(";" * 50)  # Separator for readability
        return df
    
    except Exception as e:
        print(f"Read CSV Error: {e}")

def mergePlots(sourceFolder, destinationFileName, horizontalFlag):
    images = []
    for file in sorted(os.listdir(sourceFolder)):
        if file.endswith(".png"):
            img = Image.open(os.path.join(sourceFolder, file))
            images.append(img)
    
    # Calculate total size (assuming you want a vertical merge)
    total_height = sum(img.height for img in images)
    max_width = max(img.width for img in images)
    if horizontalFlag==True:
        total_width = sum(img.width for img in images)
        max_height = max(img.height for img in images)

    # Create a blank canvas with the total size
    merged_image = Image.new('RGB', (max_width, total_height))
    if horizontalFlag==True:
        merged_image = Image.new('RGB', (total_width, max_height))

    y_offset = 0
    x_offset = 0
    for img in images:
        if horizontalFlag==True:
            merged_image.paste(img, (x_offset, 0))
            x_offset += img.width
        else:
            merged_image.paste(img, (0, y_offset))
            y_offset += img.height
    
    # Save the merged image
    merged_image.save(destinationFileName, format='PNG')

def plotRfResults(
        model, 
        X,
        y_test,
        y_pred
        ):
    # Plot Actual vs Predicted values
    plt.figure(figsize=(10, 5))
    plt.scatter(y_test, y_pred, alpha=0.7, color='blue', label="Predictions")
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r', label="Perfect Fit")
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.title("Actual vs Predicted Values")
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(plotsFolder, "RandomForest_actualVsPredicted.png"))  # Save plot
    plt.close()

    # Plot Feature Importance
    feature_importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
    plt.figure(figsize=(10, 5))
    sns.barplot(x=feature_importances, y=feature_importances.index, palette="viridis")
    plt.xlabel("Feature Importance Score")
    plt.ylabel("Features")
    plt.title("Feature Importance from Random Forest")
    plt.savefig(os.path.join(plotsFolder, "RandomForest_featureImportance.png"))  # Save plot
    plt.close()

    # Residuals Plot (Errors)
    residuals = y_test - y_pred
    plt.figure(figsize=(10, 5))
    sns.histplot(residuals, kde=True, bins=30, color='blue')
    plt.axvline(x=0, color='red', linestyle='dashed', label="Zero Error Line")
    plt.xlabel("Residuals")
    plt.title("Residuals Distribution")
    plt.legend()
    plt.savefig(os.path.join(plotsFolder, "RandomForest_residualsDistribution.png"))  # Save plot
    plt.close()

    # Time Series Line Plot: Actual vs Predicted
    plt.figure(figsize=(12, 6))
    plt.plot(y_test.values, label="Actual", color='blue', alpha=0.7)
    plt.plot(y_pred, label="Predicted", color='red', linestyle="dashed", alpha=0.7)
    plt.xlabel("Time (Index of Test Set)")
    plt.ylabel("Target Value")
    plt.title("Time Series Prediction: Actual vs Predicted")
    plt.legend()
    plt.savefig(os.path.join(plotsFolder, "RandomForest_timeSeriesPredictions.png"))  # Save plot
    plt.close()

def plotPrResults(
        y_test,
        y_pred
        ):
    # Plot Actual vs Predicted values
    plt.figure(figsize=(10, 5))
    plt.scatter(y_test, y_pred, alpha=0.7, color='blue', label="Predictions")
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r', label="Perfect Fit")
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.title("Actual vs Predicted Values (Polynomial Regression)")
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(plotsFolder, "PolynomialRegression_actualVsPredicted.png"))  # Save plot
    plt.close()

    # Residuals Plot (Errors)
    residuals = y_test - y_pred
    plt.figure(figsize=(10, 5))
    sns.histplot(residuals, kde=True, bins=30, color='blue')
    plt.axvline(x=0, color='red', linestyle='dashed', label="Zero Error Line")
    plt.xlabel("Residuals")
    plt.title("Residuals Distribution (Polynomial Regression)")
    plt.legend()
    plt.savefig(os.path.join(plotsFolder, "PolynomialRegression_residualsDistribution.png"))  # Save plot
    plt.close()

    # Time Series Line Plot: Actual vs Predicted
    plt.figure(figsize=(12, 6))
    plt.plot(y_test.values, label="Actual", color='blue', alpha=0.7)
    plt.plot(y_pred, label="Predicted", color='red', linestyle="dashed", alpha=0.7)
    plt.xlabel("Time (Index of Test Set)")
    plt.ylabel("Target Value")
    plt.title("Time Series Prediction: Actual vs Predicted (Polynomial Regression)")
    plt.legend()
    plt.savefig(os.path.join(plotsFolder, "PolynomialRegression_timeSeriesPredictions.png"))  # Save plot
    plt.close()

def plotXgbResults(
        model,
        X,
        y_test,
        y_pred
        ):
    # Plot Actual vs Predicted values
    plt.figure(figsize=(10, 5))
    plt.scatter(y_test, y_pred, alpha=0.7, color='blue', label="Predictions")
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r', label="Perfect Fit")
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.title("Actual vs Predicted Values (XGBoost)")
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(plotsFolder, "XGB_actualVsPredicted.png"))  # Save plot
    plt.close()

    # Plot Feature Importance
    feature_importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
    plt.figure(figsize=(10, 5))
    sns.barplot(x=feature_importances, y=feature_importances.index, palette="viridis")
    plt.xlabel("Feature Importance Score")
    plt.ylabel("Features")
    plt.title("Feature Importance from XGBoost")
    plt.savefig(os.path.join(plotsFolder, "XGB_featureImportance.png"))  # Save plot
    plt.close()

    # Residuals Plot (Errors)
    residuals = y_test - y_pred
    plt.figure(figsize=(10, 5))
    sns.histplot(residuals, kde=True, bins=30, color='blue')
    plt.axvline(x=0, color='red', linestyle='dashed', label="Zero Error Line")
    plt.xlabel("Residuals")
    plt.title("Residuals Distribution (XGBoost)")
    plt.legend()
    plt.savefig(os.path.join(plotsFolder, "XGB_residualsDistribution.png"))  # Save plot
    plt.close()

    # Time Series Line Plot: Actual vs Predicted
    plt.figure(figsize=(12, 6))
    plt.plot(y_test.values, label="Actual", color='blue', alpha=0.7)
    plt.plot(y_pred, label="Predicted", color='red', linestyle="dashed", alpha=0.7)
    plt.xlabel("Time (Index of Test Set)")
    plt.ylabel("Target Value")
    plt.title("Time Series Prediction: Actual vs Predicted (XGBoost)")
    plt.legend()
    plt.savefig(os.path.join(plotsFolder, "XGB_timeSeriesPredictions.png"))  # Save plot
    plt.close()

def plotAnnResults(y_test, y_pred, history, fold_split_number:int = 0):
    # Plot Actual vs Predicted values
    plt.figure(figsize=(10, 5))
    plt.scatter(y_test, y_pred, alpha=0.7, color='blue', label="Predictions")
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r', label="Perfect Fit")
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.title(f"Actual vs Predicted Values (ANN Regression) [Fold {fold_split_number}]")
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(plotsFolder, f"ANN_actualVsPredicted_{fold_split_number}.png"))  # Save plot
    plt.close()

    # Loss Curve (Training vs Validation Loss)
    plt.figure(figsize=(10, 5))
    plt.plot(history['loss'], label="Training Loss", color='blue')
    plt.plot(history['val_loss'], label="Validation Loss", color='red')
    plt.xlabel("Epochs")
    plt.ylabel("Loss (MSE)")
    plt.title(f"Loss Curve (Training vs Validation) - ANN Regression - Fold {fold_split_number}")
    plt.legend()
    plt.savefig(os.path.join(plotsFolder, f"ANN_lossCurve_{fold_split_number}.png"))  # Save plot
    plt.close()

    # Residuals Plot (Errors)
    residuals = y_test - y_pred
    plt.figure(figsize=(10, 5))
    sns.histplot(residuals, kde=True, bins=30, color='blue')
    plt.axvline(x=0, color='red', linestyle='dashed', label="Zero Error Line")
    plt.xlabel("Residuals")
    plt.title(f"Residuals Distribution (ANN Regression) [Fold {fold_split_number}]")
    plt.legend()
    plt.savefig(os.path.join(plotsFolder, f"ANN_residualsDistribution_{fold_split_number}.png"))  # Save plot
    plt.close()

    # Time Series Line Plot: Actual vs Predicted
    plt.figure(figsize=(12, 6))
    plt.plot(y_test, label="Actual", color='blue', alpha=0.7)
    plt.plot(y_pred, label="Predicted", color='red', linestyle="dashed", alpha=0.7)
    plt.xlabel("Time (Index of Test Set)")
    plt.ylabel("Target Value")
    plt.title(f"Time Series Prediction: Actual vs Predicted (ANN Regression) [Fold {fold_split_number}]")
    plt.legend()
    plt.savefig(os.path.join(plotsFolder, f"ANN_timeSeriesPredictions_{fold_split_number}.png"))  # Save plot
    plt.close()

# Merge All Data from All regions
# finalDestination = os.path.join(mergedPlots,'ElectricityGraphs.png')
# mergePlots(mergedPlots,finalDestination, True)
