#[cfg(test)]
mod caching {
    use taffy::prelude::*;
    use taffy_test_helpers::{new_test_tree, test_measure_function, TestNodeContext};

    const NODE_CONTEXT: TestNodeContext = TestNodeContext::fixed(50.0, 50.0);

    #[test]
    fn measure_count_flexbox() {
        let mut taffy = new_test_tree();

        let leaf = taffy.new_leaf_with_context(Style::default(), NODE_CONTEXT).unwrap();

        let mut node = taffy.new_with_children(Style::DEFAULT, &[leaf]).unwrap();
        for _ in 0..100 {
            node = taffy.new_with_children(Style::DEFAULT, &[node]).unwrap();
        }

        taffy.compute_layout_with_measure(node, Size::MAX_CONTENT, test_measure_function).unwrap();

        // 7 before the validity test was added to `Cache::get`, 6 after.
        //
        // These assert how often the leaf is *measured*, which is the number
        // this cache work exists to move, so the drop is the result rather than
        // a regression. Layout output is unchanged: the 5,525 generated tests
        // cover that, and they pass.
        //
        // Kept as an exact count rather than an upper bound, deliberately. A
        // `<= 7` here would stay green if the cache silently got worse again,
        // which is the failure this number is here to catch.
        assert_eq!(taffy.get_node_context_mut(leaf).unwrap().count, 6);
    }

    #[test]
    #[cfg(feature = "grid")]
    fn measure_count_grid() {
        let mut taffy = new_test_tree();

        let style = || Style { display: Display::Grid, ..Default::default() };
        let leaf = taffy.new_leaf_with_context(style(), NODE_CONTEXT).unwrap();

        let mut node = taffy.new_with_children(Style::DEFAULT, &[leaf]).unwrap();
        for _ in 0..100 {
            node = taffy.new_with_children(Style::DEFAULT, &[node]).unwrap();
        }

        taffy.compute_layout_with_measure(node, Size::MAX_CONTENT, test_measure_function).unwrap();
        // 7 before the validity test was added to `Cache::get`, 6 after.
        //
        // These assert how often the leaf is *measured*, which is the number
        // this cache work exists to move, so the drop is the result rather than
        // a regression. Layout output is unchanged: the 5,525 generated tests
        // cover that, and they pass.
        //
        // Kept as an exact count rather than an upper bound, deliberately. A
        // `<= 7` here would stay green if the cache silently got worse again,
        // which is the failure this number is here to catch.
        assert_eq!(taffy.get_node_context_mut(leaf).unwrap().count, 6);
    }
}
