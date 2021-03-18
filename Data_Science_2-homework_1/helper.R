library(data.tree)

createDataTree <- function(h2oTree) {
    h2oTreeRoot = h2oTree@root_node
    dataTree = Node$new(h2oTreeRoot@split_feature)
    dataTree$type = 'split'
    addChildren(dataTree, h2oTreeRoot)
    return(dataTree)
}

addChildren <- function(dtree, node) {
    
    if(class(node)[1] != 'H2OSplitNode') return(TRUE)
    
    feature = node@split_feature
    id = node@id
    na_direction = node@na_direction
    
    if(is.na(node@threshold)) {
        leftEdgeLabel = printValues(node@left_levels, 
                                    na_direction=='LEFT', 4)
        rightEdgeLabel = printValues(node@right_levels, 
                                     na_direction=='RIGHT', 4)
    }else {
        leftEdgeLabel = paste("<", node@threshold, 
                              ifelse(na_direction=='LEFT',',NA',''))
        rightEdgeLabel = paste(">=", node@threshold, 
                               ifelse(na_direction=='RIGHT',',NA',''))
    }
    
    left_node = node@left_child
    right_node = node@right_child
    
    if(class(left_node)[[1]] == 'H2OLeafNode')
        leftLabel = paste("prediction:", left_node@prediction)
    else
        leftLabel = left_node@split_feature
    
    if(class(right_node)[[1]] == 'H2OLeafNode')
        rightLabel = paste("prediction:", right_node@prediction)
    else
        rightLabel = right_node@split_feature
    
    if(leftLabel == rightLabel) {
        leftLabel = paste(leftLabel, "(L)")
        rightLabel = paste(rightLabel, "(R)")
    }
    
    dtreeLeft = dtree$AddChild(leftLabel)
    dtreeLeft$edgeLabel = leftEdgeLabel
    dtreeLeft$type = ifelse(class(left_node)[1] == 'H2OSplitNode', 'split', 'leaf')
    
    dtreeRight = dtree$AddChild(rightLabel)
    dtreeRight$edgeLabel = rightEdgeLabel
    dtreeRight$type = ifelse(class(right_node)[1] == 'H2OSplitNode', 'split', 'leaf')
    
    addChildren(dtreeLeft, left_node)
    addChildren(dtreeRight, right_node)
    
    return(FALSE)
}

printValues <- function(values, is_na_direction, n=4) {
    l = length(values)
    if(l == 0)
        value_string = ifelse(is_na_direction, "NA", "")
    else
        value_string = paste0(paste0(values[1:min(n,l)], collapse = ', '),
                              ifelse(l > n, ",...", ""),
                              ifelse(is_na_direction, ", NA", ""))
    return(value_string)
}


get_scores <- function (model){
    rmse <- h2o.rmse(h2o.performance(model, xval = T))
    auc <- h2o.auc(h2o.performance(model,xval = T))
    accuracy <- max(h2o.accuracy(h2o.performance(model, xval = T)))
    return(list('rmse' = rmse, 'auc' = auc, 'accuracy' = accuracy))
}  

getPerformanceMetrics <- function(model, newdata = NULL, xval = FALSE) {
    h2o.performance(model, newdata = newdata, xval = xval)@metrics$thresholds_and_metric_scores %>%
        as_tibble() %>%
        mutate(model = model@model_id)
}

plotROC <- function(performance_df) {
    ggplot(performance_df, aes(fpr, tpr, color = model)) +
        geom_path() +
        geom_abline(slope = 1, intercept = 0, linetype = "dashed") +
        coord_fixed() +
        labs(x = "False Positive Rate", y = "True Positive Rate")
}

plotRP <- function(performance_df) {
    ggplot(performance_df, aes(precision, tpr, color = model)) +  # tpr = recall
        geom_line() +
        labs(x = "Precision", y = "Recall (TPR)")
}